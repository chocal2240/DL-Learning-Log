import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

from common import (
    LABEL_NAMES,
    build_trainer,
    build_training_arguments,
    get_cache_dir,
    load_config,
    load_processed_dataset,
    resolve_storage_path,
    save_json,
)


STRATEGIES = ("random_init", "frozen", "lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one BERT medical intent classification strategy."
    )
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_model(
    strategy: str,
    config: Dict,
) -> Tuple[torch.nn.Module, Dict[str, int]]:
    model_name = config["model"]["name"]
    cache_dir = get_cache_dir(config)
    id2label = {index: label for index, label in enumerate(LABEL_NAMES)}
    label2id = {label: index for index, label in id2label.items()}
    model_kwargs = {
        "num_labels": len(LABEL_NAMES),
        "id2label": id2label,
        "label2id": label2id,
    }

    if strategy == "random_init":
        model_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        model = AutoModelForSequenceClassification.from_config(model_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )

    if strategy == "frozen":
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    elif strategy == "lora":
        strategy_config = config["strategies"]["lora"]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=strategy_config["r"],
            lora_alpha=strategy_config["alpha"],
            lora_dropout=strategy_config["dropout"],
            target_modules=strategy_config["target_modules"],
            modules_to_save=["classifier"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_stats = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
    }
    return model, parameter_stats


def compute_metrics(eval_prediction) -> Dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def initialize_wandb(
    config: Dict,
    strategy: str,
    run_name: str,
    wandb_mode: str,
    parameter_stats: Dict[str, int],
):
    if wandb_mode == "disabled":
        return None

    import wandb

    wandb_dir = resolve_storage_path(config, "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        name=run_name,
        group=config["experiment"]["name"],
        job_type="train",
        mode=wandb_mode,
        dir=str(wandb_dir),
        config={
            "strategy": strategy,
            "model": config["model"],
            "training": config["training"],
            "strategy_config": config["strategies"][strategy],
            "data": config["data"],
            **parameter_stats,
        },
    )
    if wandb_mode == "online":
        artifact_name = config["data"]["artifact_name"]
        try:
            run.use_artifact(f"{artifact_name}:latest")
        except Exception as error:
            print(
                "Warning: dataset artifact could not be attached to this run. "
                f"Run prepare_data.py --upload-to-wandb first. Details: {error}"
            )
    return run


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    strategy_config = config["strategies"][args.strategy]
    training_config = config["training"]
    seed = config["experiment"]["seed"]
    set_seed(seed)

    data_dir = resolve_storage_path(config, config["data"]["output_dir"])
    dataset = load_processed_dataset(
        data_dir,
        cache_dir=get_cache_dir(config),
    )
    dataset = dataset.rename_column("label", "labels")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        cache_dir=get_cache_dir(config),
    )

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["model"]["max_length"],
        )

    removable_columns = [
        column
        for column in dataset["train"].column_names
        if column != "labels"
    ]
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=removable_columns,
        desc="Tokenizing KUAKE-QIC",
    )

    model, parameter_stats = build_model(args.strategy, config)
    print(
        f"{args.strategy}: {parameter_stats['trainable_parameters']:,} / "
        f"{parameter_stats['total_parameters']:,} trainable parameters "
        f"({parameter_stats['trainable_percent']:.4f}%)."
    )

    run_name = args.run_name or f"{args.strategy}-seed{seed}"
    wandb_mode = args.wandb_mode or config["wandb"]["mode"]
    wandb_run = initialize_wandb(
        config=config,
        strategy=args.strategy,
        run_name=run_name,
        wandb_mode=wandb_mode,
        parameter_stats=parameter_stats,
    )

    output_dir = resolve_storage_path(config, f"outputs/{args.strategy}")
    use_fp16 = (
        training_config["fp16"]
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 7
    )
    training_arguments = build_training_arguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_config["train_batch_size"],
        per_device_eval_batch_size=training_config["eval_batch_size"],
        gradient_accumulation_steps=training_config[
            "gradient_accumulation_steps"
        ],
        learning_rate=strategy_config["learning_rate"],
        num_train_epochs=strategy_config["num_train_epochs"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=training_config["logging_steps"],
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb" if wandb_run is not None else "none",
        run_name=run_name,
        fp16=use_fp16,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )

    callbacks = []
    patience = training_config["early_stopping_patience"]
    if patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=patience)
        )

    trainer = build_trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        tokenizer=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    train_output = trainer.train()
    train_seconds = time.perf_counter() - start_time

    prediction_output = trainer.predict(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    labels = prediction_output.label_ids
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
    ).tolist()

    results = {
        "strategy": args.strategy,
        "run_name": run_name,
        "seed": seed,
        **parameter_stats,
        "train_seconds": train_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_output.metrics,
        "test_metrics": prediction_output.metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    results_path = resolve_storage_path(
        config,
        f"results/{args.strategy}.json",
    )
    save_json(results, results_path)

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    if wandb_run is not None:
        import wandb

        wandb_run.summary.update(
            {
                **parameter_stats,
                "train_seconds": train_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                **prediction_output.metrics,
            }
        )
        wandb_run.log(
            {
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels,
                    preds=predictions,
                    class_names=LABEL_NAMES,
                )
            }
        )
        wandb_run.finish()

    print(json.dumps(results["test_metrics"], ensure_ascii=False, indent=2))
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
