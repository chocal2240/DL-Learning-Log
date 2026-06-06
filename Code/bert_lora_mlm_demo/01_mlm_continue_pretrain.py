import argparse
import inspect
import math
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_NAME = "google-bert/bert-base-chinese"
DEFAULT_DATASET_NAME = "lansinuote/ChnSentiCorp"
DEFAULT_OUTPUT_DIR = "./bert-mlm-continued"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue pretraining Chinese BERT with masked language modeling."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=3000)
    parser.add_argument("--max_eval_samples", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--logging_steps", type=int, default=50)
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


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name)
    train_dataset = maybe_limit_dataset(
        dataset["train"],
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    valid_dataset = maybe_limit_dataset(
        dataset["validation"],
        max_samples=args.max_eval_samples,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_train = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing MLM texts",
    )
    tokenized_valid = valid_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=valid_dataset.column_names,
        desc="Tokenizing MLM validation texts",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=1,
        report_to=args.report_to,
        run_name=args.run_name,
        fp16=use_fp16,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    eval_result["perplexity"] = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print("MLM validation result:")
    print(eval_result)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    output_path = os.path.abspath(args.output_dir)
    print(f"MLM continued pretraining saved to: {output_path}")


if __name__ == "__main__":
    main()
