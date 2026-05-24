"""Project configuration for the Transformer practice project.

The module keeps the original lowercase names so the rest of the training
scripts can continue to import ``config.d_model`` and friends.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

# Model architecture.
d_model = 512
n_heads = 8
n_layers = 6
d_k = d_model // n_heads
d_v = d_model // n_heads
d_ff = 2048
dropout = 0.1

# Vocabulary and special token ids. These ids match the SentencePiece training
# options in tokenizer/tokenize.py.
src_vocab_size = 32000
tgt_vocab_size = 32000
padding_idx = 0
bos_idx = 2
eos_idx = 3

# Training. The default batch size is conservative for 8GB laptop GPUs.
batch_size = int(os.getenv("BATCH_SIZE", "8"))
epoch_num = int(os.getenv("EPOCH_NUM", "3"))
lr = 3e-4
train_subset_size = int(os.getenv("TRAIN_SUBSET_SIZE", "0"))
dev_subset_size = int(os.getenv("DEV_SUBSET_SIZE", "0"))

# Weights & Biases logging. Keep disabled by default so local training does not
# prompt for credentials unless explicitly requested.
use_wandb = os.getenv("USE_WANDB", "0").lower() in {"1", "true", "yes", "on"}
wandb_project = os.getenv("WANDB_PROJECT", "transformer-practice")
wandb_entity = os.getenv("WANDB_ENTITY") or None
wandb_run_name = os.getenv("WANDB_RUN_NAME") or None
wandb_log_interval = max(1, int(os.getenv("WANDB_LOG_INTERVAL", "50")))
wandb_watch = os.getenv("WANDB_WATCH", "0").lower() in {"1", "true", "yes", "on"}
wandb_log_model = os.getenv("WANDB_LOG_MODEL", "0").lower() in {"1", "true", "yes", "on"}

# Decoding.
max_len = 60
beam_size = 3

# Paths.
data_dir = PROJECT_ROOT / "data"
train_data_path = data_dir / "json" / "train.json"
dev_data_path = data_dir / "json" / "dev.json"
test_data_path = data_dir / "json" / "test.json"

model_path = PROJECT_ROOT / "weights" / "transformer_model.pth"
test_model_path = PROJECT_ROOT / "run" / "train" / "exp" / "weights" / "best_bleu_26.30.pth"

# Device selection. CUDA is used when available; otherwise everything falls
# back to CPU so the scripts remain runnable on laptops without an NVIDIA GPU.
gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0")
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
device_id = list(range(torch.cuda.device_count())) if cuda_available else []
