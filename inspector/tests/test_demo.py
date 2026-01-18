"""Integration smoke test for the inspector on ResNet18."""
from __future__ import annotations

import pathlib
import sys

import torch
from torchvision.models import resnet18

# Ensure the src/ directory is importable when running tests directly.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from patch_utils import enable_dev_mode
from inspector.renderers import MermaidRenderer


def test_demo_mermaid_export() -> None:
    enable_dev_mode()

    model = resnet18(weights=None)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    with model.inspect() as inspector:
        y = model(dummy)
        loss = y.sum()
        loss.backward()

    renderer = MermaidRenderer(show_legend=True, detail_level="none")
    mermaid = inspector.export(renderer, path="out/graph.mmd")
    print("\n" + mermaid)

    assert "graph TD" in mermaid
