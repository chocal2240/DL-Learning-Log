"""Single-sentence translation helper."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch

import config
from beam_decoder import beam_search
from model.tf_model import Transformer, make_model
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s:%(lineno)d",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def load_checkpoint(model: Transformer, checkpoint_path: str | Path = config.test_model_path) -> Transformer:
    """Load model weights onto the configured device."""

    state_dict = torch.load(Path(checkpoint_path), map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def translate(src: torch.Tensor, model: Transformer) -> str:
    """Translate a batch containing one source sentence tensor."""

    tokenizer = chinese_tokenizer_load()
    src_mask = (src != config.padding_idx).unsqueeze(-2)

    with torch.no_grad():
        decode_result, _ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,
            config.padding_idx,
            config.bos_idx,
            config.eos_idx,
            config.beam_size,
            config.device,
        )

    best_tokens = decode_result[0][0]
    return tokenizer.decode(best_tokens)


def build_model() -> Transformer:
    return make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_heads,
        config.dropout,
    )


def one_sentence_translate(sentence: str, checkpoint_path: str | Path = config.test_model_path) -> str:
    """Translate one English sentence into Chinese."""

    model = load_checkpoint(build_model(), checkpoint_path)
    tokenizer = english_tokenizer_load()
    token_ids = [
        config.bos_idx,
        *tokenizer.encode(sentence, out_type=int),
        config.eos_idx,
    ]
    batch_input = torch.tensor([token_ids], dtype=torch.long, device=config.device)
    return translate(batch_input, model)


def translate_example() -> None:
    """Interactive translation loop."""

    LOGGER.info("Example: The government has implemented various policies to improve living standards.")

    while True:
        sentence = input("请输入英文句子进行翻译：").strip()
        if not sentence:
            continue
        print("翻译结果：", one_sentence_translate(sentence))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    translate_example()
