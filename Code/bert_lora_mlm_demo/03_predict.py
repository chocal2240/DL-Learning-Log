import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_BASE_MODEL = "./bert-mlm-continued"
DEFAULT_LORA_MODEL = "./bert-lora-sentiment"

ID2LABEL = {
    0: "负面",
    1: "正面",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sentiment prediction with a BERT LoRA adapter."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_model", default=DEFAULT_LORA_MODEL)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--text",
        action="append",
        help="Input text. Use this option multiple times for multiple examples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    texts = args.text or [
        "这个酒店位置很好，房间也很干净，下次还会再来。",
        "服务态度太差了，房间也很脏，完全不推荐。",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.lora_model)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    model = PeftModel.from_pretrained(base_model, args.lora_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    for text, pred, prob in zip(texts, preds, probs):
        print("=" * 50)
        print("文本：", text)
        print("预测：", ID2LABEL[pred.item()])
        print("概率：", [round(x, 4) for x in prob.detach().cpu().tolist()])


if __name__ == "__main__":
    main()
