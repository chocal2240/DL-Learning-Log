"""Dataset and mask utilities for machine translation training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import config
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask future positions for autoregressive decoder self-attention."""

    return torch.tril(torch.ones(1, size, size, dtype=torch.bool))


class Batch:
    """A batch of token ids, raw text, and the masks used by the Transformer."""

    def __init__(
        self,
        src_text: list[str],
        trg_text: list[str],
        src: torch.Tensor,
        trg: torch.Tensor | None = None,
        pad: int = config.padding_idx,
    ) -> None:
        self.src_text = src_text
        self.trg_text = trg_text
        self.src = src.to(config.device)
        self.src_mask = (self.src != pad).unsqueeze(-2)

        if trg is None:
            self.trg = None
            self.trg_y = None
            self.trg_mask = None
            self.ntokens = torch.tensor(0, device=config.device)
            return

        trg = trg.to(config.device)
        self.trg = trg[:, :-1]
        self.trg_y = trg[:, 1:]
        self.trg_mask = self.make_std_mask(self.trg, pad)
        self.ntokens = (self.trg_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """Create a target mask that hides padding and future positions."""

        tgt_mask = (tgt != pad).unsqueeze(-2)
        future_mask = subsequent_mask(tgt.size(-1)).to(device=tgt.device)
        return tgt_mask & future_mask


class MTDataset(Dataset):
    """Machine translation dataset backed by ``[[english, chinese], ...]`` JSON."""

    def __init__(self, data_path: str | Path, sort: bool = True) -> None:
        self.examples = self._load_examples(data_path)
        if sort:
            self.examples.sort(key=lambda pair: len(pair[0]))

        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.pad_id = self.sp_eng.pad_id()

    @staticmethod
    def _load_examples(data_path: str | Path) -> list[tuple[str, str]]:
        path = Path(data_path)
        with path.open("r", encoding="utf-8") as file:
            raw_examples = json.load(file)

        return [(str(en), str(zh)) for en, zh in raw_examples]

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)

    def collate_fn(self, batch: Sequence[tuple[str, str]]) -> Batch:
        src_text = [source for source, _ in batch]
        tgt_text = [target for _, target in batch]

        src_tokens = [self._encode(self.sp_eng, sentence) for sentence in src_text]
        tgt_tokens = [self._encode(self.sp_chn, sentence) for sentence in tgt_text]

        src_tensor = self._pad_tokens(src_tokens)
        tgt_tensor = self._pad_tokens(tgt_tokens)
        return Batch(src_text, tgt_text, src_tensor, tgt_tensor, self.pad_id)

    def _encode(self, tokenizer, sentence: str) -> list[int]:
        return [tokenizer.bos_id(), *tokenizer.encode(sentence, out_type=int), tokenizer.eos_id()]

    def _pad_tokens(self, token_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        tensors = [torch.tensor(ids, dtype=torch.long) for ids in token_ids]
        return pad_sequence(tensors, batch_first=True, padding_value=self.pad_id)
