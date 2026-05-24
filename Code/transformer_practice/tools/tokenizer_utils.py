"""Helpers for loading SentencePiece tokenizers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import sentencepiece as spm

import config


TOKENIZER_DIR = config.PROJECT_ROOT / "tokenizer"


@lru_cache(maxsize=None)
def load_tokenizer(model_path: str | Path) -> spm.SentencePieceProcessor:
    """Load and cache a SentencePiece model."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {path}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(path))
    return tokenizer


def chinese_tokenizer_load() -> spm.SentencePieceProcessor:
    """Return the cached Chinese tokenizer."""

    return load_tokenizer(TOKENIZER_DIR / "chn.model")


def english_tokenizer_load() -> spm.SentencePieceProcessor:
    """Return the cached English tokenizer."""

    return load_tokenizer(TOKENIZER_DIR / "eng.model")
