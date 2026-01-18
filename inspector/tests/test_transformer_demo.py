"""Integration test on PyTorch Transformer encoder for Inspector."""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

# Ensure the src/ directory is importable when running tests directly.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from patch_utils import enable_dev_mode
from inspector.renderers import MermaidRenderer


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        return self.encoder(x)


def test_transformer_mermaid_export() -> None:
    enable_dev_mode()

    model = TinyTransformer().train()
    src = torch.randn(12, 2, 32)  # seq_len, batch, d_model

    with model.inspect() as inspector:
        out = model(src)
        loss = out.sum()
        loss.backward()

    # Show full graph to ensure attention/FFN nodes are present
    renderer = MermaidRenderer(detail_level="compact", collapse_depth=None, show_legend=True)
    mermaid = inspector.export(renderer, path="out/transformer.mmd")
    print("\n" + mermaid)

    assert "graph TD" in mermaid
    # Expect feedforward components to appear; attention uses functional path in PyTorch and may not surface as a leaf module.
    assert "linear1" in mermaid
    assert "linear2" in mermaid
    assert "norm1" in mermaid and "norm2" in mermaid
