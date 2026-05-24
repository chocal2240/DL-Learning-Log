"""Training and evaluation entry point."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model.tf_model import Transformer, make_model
from model.train_utils import MultiGPULossCompute, get_std_opt
from tools.create_exp_folder import create_exp_folder
from tools.data_loader import MTDataset
from tools.tokenizer_utils import chinese_tokenizer_load

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for local training.
    wandb = None


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def run_epoch(
    data: DataLoader,
    model: torch.nn.Module,
    loss_compute: MultiGPULossCompute,
    *,
    phase: str,
    epoch: int,
    wandb_run: Any | None = None,
    global_step: int = 0,
) -> tuple[float, int]:
    """Run one full pass over a dataloader and return average loss per token."""

    total_tokens = 0
    total_loss = 0.0

    for batch_index, batch in enumerate(tqdm(data, desc=phase), start=1):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        batch_loss = loss_compute(out, batch.trg_y, batch.ntokens)
        batch_tokens = int(batch.ntokens.item())
        total_loss += batch_loss
        total_tokens += batch_tokens

        if phase == "train":
            global_step += 1
            if wandb_run is not None and batch_index % config.wandb_log_interval == 0:
                wandb_run.log(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "train/batch_loss": batch_loss / max(batch_tokens, 1),
                        "train/learning_rate": getattr(loss_compute.optimizer, "_rate", 0.0),
                        "train/tokens": batch_tokens,
                    },
                    step=global_step,
                )

    return total_loss / max(total_tokens, 1), global_step


def init_wandb(train_data: DataLoader, dev_data: DataLoader) -> Any | None:
    """Create a wandb run when enabled in config."""

    if not config.use_wandb:
        return None
    if wandb is None:
        LOGGER.warning("USE_WANDB is enabled, but wandb is not installed. Continuing without wandb.")
        return None

    cuda_device_name = None
    if config.device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(config.device.index or 0)

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        job_type="train",
        config={
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
            "src_vocab_size": config.src_vocab_size,
            "tgt_vocab_size": config.tgt_vocab_size,
            "batch_size": config.batch_size,
            "epoch_num": config.epoch_num,
            "max_len": config.max_len,
            "beam_size": config.beam_size,
            "train_examples": len(train_data.dataset),
            "dev_examples": len(dev_data.dataset),
            "train_batches": len(train_data),
            "dev_batches": len(dev_data),
            "train_subset_size": config.train_subset_size or None,
            "dev_subset_size": config.dev_subset_size or None,
            "device": str(config.device),
            "cuda_device_name": cuda_device_name,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("best/*", step_metric="epoch")

    return run


def log_checkpoint_artifact(wandb_run: Any | None, checkpoint_path: Path, aliases: list[str]) -> None:
    """Upload a checkpoint as a wandb artifact when model logging is enabled."""

    if wandb_run is None or wandb is None or not config.wandb_log_model:
        return

    artifact = wandb.Artifact(f"{wandb_run.id}-{checkpoint_path.stem}", type="model")
    artifact.add_file(str(checkpoint_path))
    wandb_run.log_artifact(artifact, aliases=aliases)


def train(
    train_data: DataLoader,
    dev_data: DataLoader,
    model: Transformer,
    train_model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer,
    wandb_run: Any | None = None,
) -> None:
    """Train the model and save best/last checkpoints."""

    best_bleu_score = float("-inf")
    exp_folder, weights_folder = create_exp_folder()
    if wandb_run is not None:
        wandb_run.config.update({"exp_folder": str(exp_folder)}, allow_val_change=True)

    global_step = 0

    for epoch in range(1, config.epoch_num + 1):
        if config.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(config.device)

        LOGGER.info("Epoch %s/%s: training", epoch, config.epoch_num)
        train_model.train()
        train_loss, global_step = run_epoch(
            train_data,
            train_model,
            MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer),
            phase="train",
            epoch=epoch,
            wandb_run=wandb_run,
            global_step=global_step,
        )

        LOGGER.info("Epoch %s/%s: validating", epoch, config.epoch_num)
        train_model.eval()
        with torch.no_grad():
            dev_loss, _ = run_epoch(
                dev_data,
                train_model,
                MultiGPULossCompute(model.generator, criterion, config.device_id),
                phase="val",
                epoch=epoch,
                wandb_run=wandb_run,
                global_step=global_step,
            )
        bleu_score = evaluate(dev_data, model)

        LOGGER.info(
            "Epoch %s: train_loss=%.3f, val_loss=%.3f, BLEU=%.2f",
            epoch,
            train_loss,
            dev_loss,
            bleu_score,
        )

        epoch_metrics = {
            "global_step": global_step,
            "epoch": epoch,
            "train/loss": train_loss,
            "train/learning_rate": getattr(optimizer, "_rate", 0.0),
            "val/loss": dev_loss,
            "val/bleu": bleu_score,
        }
        if config.device.type == "cuda":
            epoch_metrics["system/cuda_peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(config.device) / (
                1024**2
            )

        if bleu_score > best_bleu_score:
            if best_bleu_score != float("-inf"):
                old_model_path = weights_folder / f"best_bleu_{best_bleu_score:.2f}.pth"
                old_model_path.unlink(missing_ok=True)

            best_model_path = weights_folder / f"best_bleu_{bleu_score:.2f}.pth"
            torch.save(model.state_dict(), best_model_path)
            best_bleu_score = bleu_score
            epoch_metrics["best/bleu"] = best_bleu_score
            if wandb_run is not None:
                wandb_run.summary["best_checkpoint"] = str(best_model_path)
            log_checkpoint_artifact(wandb_run, best_model_path, aliases=["best", f"epoch-{epoch}"])

        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=global_step)
            wandb_run.summary["best_bleu"] = best_bleu_score

        if epoch == config.epoch_num:
            last_model_path = weights_folder / f"last_bleu_{bleu_score:.2f}.pth"
            torch.save(model.state_dict(), last_model_path)
            if wandb_run is not None:
                wandb_run.summary["last_bleu"] = bleu_score
                wandb_run.summary["last_checkpoint"] = str(last_model_path)
            log_checkpoint_artifact(wandb_run, last_model_path, aliases=["last", f"epoch-{epoch}"])


def evaluate(data: DataLoader, model: Transformer) -> float:
    """Translate a dataset and return corpus BLEU."""

    tokenizer = chinese_tokenizer_load()
    references: list[str] = []
    hypotheses: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            decode_result, _ = beam_search(
                model,
                batch.src,
                batch.src_mask,
                config.max_len,
                config.padding_idx,
                config.bos_idx,
                config.eos_idx,
                config.beam_size,
                config.device,
            )
            best_hypotheses = [hypotheses_for_sentence[0] for hypotheses_for_sentence in decode_result]
            translations = [tokenizer.decode(token_ids) for token_ids in best_hypotheses]
            references.extend(batch.trg_text)
            hypotheses.extend(translations)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
    return float(bleu.score)


def test(data: DataLoader, model: Transformer, criterion: torch.nn.Module, model_path: str | Path | None = None) -> None:
    """Load a checkpoint, evaluate test loss, and report BLEU."""

    checkpoint_path = Path(model_path or config.test_model_path)
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    with torch.no_grad():
        loss, _ = run_epoch(
            data,
            model,
            MultiGPULossCompute(model.generator, criterion, config.device_id),
            phase="test",
            epoch=0,
        )
    bleu_score = evaluate(data, model)
    LOGGER.info("Test loss=%.3f, BLEU=%.2f", loss, bleu_score)


def build_dataloader(data_path: str | Path, shuffle: bool, limit: int = 0) -> DataLoader:
    base_dataset = MTDataset(data_path)
    dataset = base_dataset
    if limit > 0:
        dataset = Subset(base_dataset, range(min(limit, len(base_dataset))))

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        collate_fn=base_dataset.collate_fn,
    )


def build_model() -> Transformer:
    return make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_heads,
        config.dropout,
    )


def parallelize_model(model: Transformer) -> torch.nn.Module:
    if config.device.type == "cuda" and len(config.device_id) > 1:
        return torch.nn.DataParallel(model, device_ids=config.device_id)
    return model


def run() -> None:
    LOGGER.info("Using device: %s", config.device)
    if config.device.type == "cuda":
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(config.device.index or 0))

    train_dataloader = build_dataloader(config.train_data_path, shuffle=True, limit=config.train_subset_size)
    dev_dataloader = build_dataloader(config.dev_data_path, shuffle=False, limit=config.dev_subset_size)
    wandb_run = init_wandb(train_dataloader, dev_dataloader)

    model = build_model()
    train_model = parallelize_model(model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_idx, reduction="sum")
    optimizer = get_std_opt(model)

    if config.wandb_watch and wandb_run is not None and wandb is not None:
        wandb.watch(model, log="gradients", log_freq=config.wandb_log_interval)

    try:
        train(train_dataloader, dev_dataloader, model, train_model, criterion, optimizer, wandb_run)
        # test_dataloader = build_dataloader(config.test_data_path, shuffle=False)
        # test(test_dataloader, model, criterion)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
