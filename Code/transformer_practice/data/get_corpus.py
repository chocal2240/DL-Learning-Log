"""Build plain-text parallel corpora from the JSON train/dev/test files."""

from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
JSON_DIR = DATA_DIR / "json"
SPLITS = ("train", "dev", "test")


def load_split(split: str) -> list[tuple[str, str]]:
    with (JSON_DIR / f"{split}.json").open("r", encoding="utf-8") as file:
        return [(str(en), str(zh)) for en, zh in json.load(file)]


def build_corpus() -> tuple[int, int]:
    english_lines: list[str] = []
    chinese_lines: list[str] = []

    for split in SPLITS:
        for english, chinese in load_split(split):
            english_lines.append(f"{english}\n")
            chinese_lines.append(f"{chinese}\n")

    (DATA_DIR / "corpus.en").write_text("".join(english_lines), encoding="utf-8")
    (DATA_DIR / "corpus.ch").write_text("".join(chinese_lines), encoding="utf-8")
    return len(english_lines), len(chinese_lines)


def main() -> None:
    english_count, chinese_count = build_corpus()
    print(f"lines of English: {english_count}")
    print(f"lines of Chinese: {chinese_count}")
    print("-------- Get Corpus ! --------")


if __name__ == "__main__":
    main()
