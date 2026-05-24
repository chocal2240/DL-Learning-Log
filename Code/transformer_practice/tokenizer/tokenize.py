"""Train SentencePiece tokenizers for English and Chinese corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    input_file: Path
    vocab_size: int
    model_prefix: Path
    model_type: str
    character_coverage: float


def train(config: TokenizerTrainingConfig) -> None:
    """Train one SentencePiece model with fixed special-token ids."""

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(config.input_file),
        model_prefix=str(config.model_prefix),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


def run() -> None:
    configs = [
        TokenizerTrainingConfig(
            input_file=DATA_DIR / "corpus.en",
            vocab_size=32_000,
            model_prefix=TOKENIZER_DIR / "eng",
            model_type="bpe",
            character_coverage=1.0,
        ),
        TokenizerTrainingConfig(
            input_file=DATA_DIR / "corpus.ch",
            vocab_size=32_000,
            model_prefix=TOKENIZER_DIR / "chn",
            model_type="bpe",
            character_coverage=0.9995,
        ),
    ]

    for tokenizer_config in configs:
        train(tokenizer_config)


def test() -> None:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(TOKENIZER_DIR / "chn.model"))

    text = "美国总统特朗普今日抵达夏威夷。"
    print(tokenizer.encode(text, out_type=str))
    print(tokenizer.encode(text, out_type=int))
    print(tokenizer.decode([12907, 277, 7419, 7318, 18384, 28724]))


if __name__ == "__main__":
    run()
    # test()
