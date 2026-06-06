import argparse
import inspect
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_NAME = "./bert-mlm-continued"
DEFAULT_DATASET_NAME = "lansinuote/ChnSentiCorp"
DEFAULT_OUTPUT_DIR = "./bert-lora-sentiment"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Chinese BERT for sentiment classification with LoRA."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable fp16 even when CUDA is available.",
    )
    return parser.parse_args()


def maybe_limit_dataset(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0:
        return dataset
    sample_count = min(max_samples, len(dataset))
    return dataset.shuffle(seed=seed).select(range(sample_count))


def build_training_arguments(**kwargs):
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters

    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in parameters:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    return TrainingArguments(**kwargs)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name)
    train_dataset = maybe_limit_dataset(
        dataset["train"],
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True, desc="Tokenizing train")
    valid_dataset = valid_dataset.map(tokenize_fn, batched=True, desc="Tokenizing valid")
    test_dataset = test_dataset.map(tokenize_fn, batched=True, desc="Tokenizing test")

    train_dataset = train_dataset.rename_column("label", "labels")
    valid_dataset = valid_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset = train_dataset.remove_columns(["text"])
    valid_dataset = valid_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        # BertForMaskedLM has no pooler, so the classification model creates
        # one. Save it with the adapter to keep inference reproducible.
        modules_to_save=["classifier", "pooler"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    train_args = build_training_arguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=2,
        report_to=args.report_to,
        run_name=args.run_name,
        fp16=use_fp16,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    test_result = trainer.evaluate(test_dataset)
    print("Test result:")
    print(test_result)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    output_path = os.path.abspath(args.output_dir)
    print(f"LoRA adapter saved to: {output_path}")


if __name__ == "__main__":
    main()
