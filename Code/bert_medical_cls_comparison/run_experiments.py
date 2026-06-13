import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from common import PROJECT_DIR, load_config, resolve_storage_path
from train import STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data, run all strategies, and summarize results."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-data-upload", action="store_true")
    return parser.parse_args()


def run_command(command: List[str]) -> None:
    print(f"\n> {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def read_results(strategies: List[str], config: Dict) -> List[Dict]:
    rows = []
    for strategy in strategies:
        path = resolve_storage_path(config, f"results/{strategy}.json")
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        test_metrics = result["test_metrics"]
        rows.append(
            {
                "strategy": strategy,
                "test_accuracy": test_metrics["test_accuracy"],
                "test_macro_f1": test_metrics["test_macro_f1"],
                "test_weighted_f1": test_metrics["test_weighted_f1"],
                "train_seconds": result["train_seconds"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                "trainable_parameters": result["trainable_parameters"],
                "trainable_percent": result["trainable_percent"],
            }
        )
    return rows


def save_summary(rows: List[Dict], config: Dict) -> Path:
    results_dir = resolve_storage_path(config, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = results_dir / "comparison.md"
    headers = list(rows[0].keys())
    with markdown_path.open("w", encoding="utf-8") as file:
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [
                f"{row[header]:.4f}"
                if isinstance(row[header], float)
                else str(row[header])
                for header in headers
            ]
            file.write("| " + " | ".join(values) + " |\n")
    return csv_path


def log_comparison_to_wandb(
    rows: List[Dict],
    config: Dict,
    wandb_mode: str,
) -> None:
    if wandb_mode == "disabled":
        return

    import wandb

    wandb_dir = resolve_storage_path(config, "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        name="strategy-comparison",
        group=config["experiment"]["name"],
        job_type="evaluation",
        mode=wandb_mode,
        dir=str(wandb_dir),
    )
    columns = list(rows[0].keys())
    table = wandb.Table(
        columns=columns,
        data=[[row[column] for column in columns] for row in rows],
    )
    run.log(
        {
            "comparison/table": table,
            "comparison/macro_f1": wandb.plot.bar(
                table,
                "strategy",
                "test_macro_f1",
                title="Test Macro-F1 by strategy",
            ),
            "comparison/train_seconds": wandb.plot.bar(
                table,
                "strategy",
                "train_seconds",
                title="Training time by strategy",
            ),
        }
    )
    run.finish()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    wandb_mode = args.wandb_mode or config["wandb"]["mode"]

    prepare_command = [
        sys.executable,
        "prepare_data.py",
        "--config",
        args.config,
        "--wandb-mode",
        wandb_mode,
    ]
    if args.force_data:
        prepare_command.append("--force")
    if not args.skip_data_upload and wandb_mode != "disabled":
        prepare_command.append("--upload-to-wandb")
    run_command(prepare_command)

    for strategy in args.strategies:
        run_command(
            [
                sys.executable,
                "train.py",
                "--config",
                args.config,
                "--strategy",
                strategy,
                "--wandb-mode",
                wandb_mode,
            ]
        )

    rows = read_results(args.strategies, config)
    csv_path = save_summary(rows, config)
    log_comparison_to_wandb(rows, config, wandb_mode)
    print(f"\nComparison saved to: {csv_path}")


if __name__ == "__main__":
    main()
