"""Print simple statistics for the generated parallel corpus files."""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def describe_file(path: Path, lines: list[str]) -> None:
    char_count = sum(len(line.strip()) for line in lines)
    average_chars = char_count / max(len(lines), 1)

    print(f"\n{path.name} ({path}):")
    print(f"  文件大小: {path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  总行数: {len(lines):,}")
    print(f"  总字符数: {char_count:,}")
    print(f"  平均每行字符数: {average_chars:.1f}")


def analyze_corpus(ch_path: str | Path, en_path: str | Path) -> None:
    chinese_path = Path(ch_path)
    english_path = Path(en_path)

    chinese_lines = read_lines(chinese_path)
    english_lines = read_lines(english_path)

    print("=" * 50)
    print("语料库统计信息")
    print("=" * 50)
    describe_file(chinese_path, chinese_lines)
    describe_file(english_path, english_lines)

    if len(chinese_lines) == len(english_lines):
        print("\n中英文文件行数匹配")
    else:
        print("\n警告：中英文文件行数不匹配！")

    print("=" * 50)


def main() -> None:
    analyze_corpus(DATA_DIR / "corpus.ch", DATA_DIR / "corpus.en")


if __name__ == "__main__":
    main()
