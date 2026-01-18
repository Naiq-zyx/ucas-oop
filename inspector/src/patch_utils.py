"""Development utility to monkey-patch torch.nn.Module.inspect."""
from __future__ import annotations

import torch

from inspector.core import Inspector


def enable_dev_mode(pattern: str = "*") -> None:
    """Enable the native inspector via monkey patching.

    This attaches an `inspect` method to `torch.nn.Module` that returns
    an `Inspector` context manager for the given model instance.
    """

    def inspect(self: torch.nn.Module, *, pattern: str = pattern) -> Inspector:
        return Inspector(self, pattern=pattern)

    torch.nn.Module.inspect = inspect  # type: ignore[attr-defined]
    print("[inspector] Development mode enabled: torch.nn.Module.inspect attached")
