import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from common import (
    LABEL_NAMES,
    get_cache_dir,
    load_config,
    resolve_storage_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small stratified KUAKE-QIC dataset."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--upload-to-wandb", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    return parser.parse_args()


def split_indices(
    labels: List[int],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")

    all_indices = list(range(len(labels)))
    train_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=validation_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    holdout_labels = [labels[index] for index in holdout_indices]
    relative_test_ratio = test_ratio / (validation_ratio + test_ratio)
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=holdout_labels,
    )
    return {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }


def normalize_source_dataset(source: Dataset) -> Dataset:
    label2id = {label: index for index, label in enumerate(LABEL_NAMES)}
    rows = {"id": [], "text": [], "label": [], "label_name": []}

    for row_index, row in enumerate(source):
        label_name = str(row["label"]).strip()
        text = str(row["query"]).strip()
        if not text or label_name not in label2id:
            continue
        rows["id"].append(str(row.get("id", row_index)))
        rows["text"].append(text)
        rows["label"].append(label2id[label_name])
        rows["label_name"].append(label_name)

    if not rows["text"]:
        raise ValueError("No valid KUAKE-QIC rows were found in the source dataset.")
    return Dataset.from_dict(rows)


def upload_artifact(
    data_dir: Path,
    metadata: Dict,
    config: Dict,
    wandb_mode: str,
) -> None:
    if wandb_mode == "disabled":
        return

    import wandb

    wandb_config = config["wandb"]
    wandb_dir = resolve_storage_path(config, "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        name="prepare-kuake-qic-data",
        job_type="prepare-data",
        mode=wandb_mode,
        dir=str(wandb_dir),
        config={"dataset": config["data"], "seed": config["experiment"]["seed"]},
    )
    artifact = wandb.Artifact(
        name=config["data"]["artifact_name"],
        type="dataset",
        description=(
            "Stratified KUAKE-QIC subset used by the BERT strategy comparison."
        ),
        metadata=metadata,
    )
    artifact.add_dir(str(data_dir))
    run.log_artifact(artifact)

    table = wandb.Table(
        columns=["split", "id", "text", "label", "label_name"]
    )
    for split in ("train", "validation", "test"):
        with (data_dir / f"{split}.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                table.add_data(
                    split,
                    row["id"],
                    row["text"],
                    row["label"],
                    row["label_name"],
                )
    run.log({"dataset/all_samples": table})
    run.finish()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    output_dir = resolve_storage_path(config, data_config["output_dir"])
    expected_files = [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "test.jsonl",
        output_dir / "metadata.json",
    ]

    if all(path.exists() for path in expected_files) and not args.force:
        print(f"Prepared data already exists: {output_dir}")
    else:
        source = load_dataset(
            data_config["dataset_name"],
            split=data_config["dataset_split"],
            cache_dir=get_cache_dir(config),
        )
        normalized = normalize_source_dataset(source)
        indices = split_indices(
            labels=normalized["label"],
            train_ratio=data_config["train_ratio"],
            validation_ratio=data_config["validation_ratio"],
            test_ratio=data_config["test_ratio"],
            seed=config["experiment"]["seed"],
        )
        dataset = DatasetDict(
            {
                split: normalized.select(split_indices)
                for split, split_indices in indices.items()
            }
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for split, split_dataset in dataset.items():
            split_dataset.to_json(
                output_dir / f"{split}.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )

        metadata = {
            "source_dataset": data_config["dataset_name"],
            "source_split": data_config["dataset_split"],
            "seed": config["experiment"]["seed"],
            "label_names": LABEL_NAMES,
            "split_sizes": {split: len(data) for split, data in dataset.items()},
            "label_distribution": {
                split: {
                    LABEL_NAMES[label_id]: count
                    for label_id, count in sorted(Counter(data["label"]).items())
                }
                for split, data in dataset.items()
            },
        }
        save_json(metadata, output_dir / "metadata.json")
        print(f"Prepared {len(normalized)} samples in: {output_dir}")
        print(metadata["split_sizes"])

    if args.upload_to_wandb:
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        upload_artifact(
            data_dir=output_dir,
            metadata=metadata,
            config=config,
            wandb_mode=args.wandb_mode or config["wandb"]["mode"],
        )


if __name__ == "__main__":
    main()
