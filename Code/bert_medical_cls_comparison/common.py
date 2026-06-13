import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from datasets import DatasetDict, load_dataset
from transformers import Trainer, TrainingArguments


PROJECT_DIR = Path(__file__).resolve().parent

LABEL_NAMES = [
    "病情诊断",
    "病因分析",
    "治疗方案",
    "就医建议",
    "指标解读",
    "疾病表述",
    "后果表述",
    "注意事项",
    "功效作用",
    "医疗费用",
    "其他",
]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def resolve_storage_path(config: Dict[str, Any], path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    storage_root = Path(
        config.get("storage", {}).get("root_dir", str(PROJECT_DIR))
    )
    return storage_root / path


def get_cache_dir(config: Dict[str, Any]) -> Optional[str]:
    cache_dir = config.get("storage", {}).get("cache_dir")
    if not cache_dir:
        return None
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_processed_dataset(
    data_dir: Path,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    files = {
        split: str(data_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    missing = [path for path in files.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared dataset is missing. Run prepare_data.py first. Missing: "
            + ", ".join(missing)
        )
    return load_dataset("json", data_files=files, cache_dir=cache_dir)


def build_training_arguments(**kwargs: Any) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if (
        "evaluation_strategy" in kwargs
        and "evaluation_strategy" not in parameters
        and "eval_strategy" in parameters
    ):
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return TrainingArguments(**supported_kwargs)


def build_trainer(**kwargs: Any) -> Trainer:
    parameters = inspect.signature(Trainer.__init__).parameters
    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is not None:
        if "processing_class" in parameters:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in parameters:
            kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
