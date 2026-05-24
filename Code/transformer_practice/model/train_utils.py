"""Training helpers: loss computation and the Noam learning-rate schedule."""

from __future__ import annotations

import torch
from torch import nn


class MultiGPULossCompute:
    """Compute token-level loss with a modern autograd path.

    The original project manually scattered tensors across GPUs. That style is
    brittle on recent PyTorch versions and unnecessary here because the model can
    already be wrapped by ``nn.DataParallel``. The class name is kept for
    backwards compatibility with the old training script.
    """

    def __init__(
        self,
        generator: nn.Module,
        criterion: nn.Module,
        devices: list[int] | None = None,
        opt: NoamOpt | torch.optim.Optimizer | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.generator = generator
        self.criterion = criterion
        self.optimizer = opt
        self.devices = devices or []
        self.chunk_size = chunk_size

    def __call__(self, out: torch.Tensor, targets: torch.Tensor, normalize: torch.Tensor | int) -> float:
        normalizer = float(normalize.item() if isinstance(normalize, torch.Tensor) else normalize)
        if normalizer == 0:
            return 0.0

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        log_probs = self.generator(out)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)),
            targets.contiguous().view(-1),
        )
        normalized_loss = loss / normalizer

        if self.optimizer is not None:
            normalized_loss.backward()
            self.optimizer.step()

        return float(loss.detach().item())


class NoamOpt:
    """Optimizer wrapper using the Transformer paper's learning-rate schedule."""

    def __init__(
        self,
        model_size: int,
        factor: float,
        warmup: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0.0

    def step(self) -> None:
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def rate(self, step: int | None = None) -> float:
        step = self._step if step is None else step
        step = max(step, 1)
        return self.factor * (
            self.model_size ** -0.5 * min(step ** -0.5, step * self.warmup ** -1.5)
        )


def get_std_opt(model: nn.Module) -> NoamOpt:
    """Return Adam with the standard Noam learning-rate schedule."""

    return NoamOpt(
        model.src_embed[0].d_model,
        factor=1.0,
        warmup=10_000,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9),
    )
