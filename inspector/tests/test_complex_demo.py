"""Integration test on a branched/reused module topology."""
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


class ComplexNet(nn.Module):
    """Branched network with shared ReLU used multiple times to test reuse/in-place."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.shared_relu = nn.ReLU(inplace=True)
        self.branch1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1),
        )
        # stride=1 & padding keep spatial dims so branch merge is valid
        self.branch2_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_conv = nn.Conv2d(16, 16, kernel_size=1)
        self.tail = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        b1 = self.branch1(x)
        b1 = self.shared_relu(b1)  # reuse 1

        b2 = self.branch2_pool(x)
        b2 = self.shared_relu(b2)  # reuse 2 (in-place)
        b2 = self.branch2_conv(b2)

        merged = b1 + b2
        merged = self.shared_relu(merged)  # reuse 3
        out = self.tail(merged)
        return out.mean(dim=(2, 3))


def test_complex_mermaid_export() -> None:
    enable_dev_mode()

    model = ComplexNet().eval()
    dummy = torch.randn(1, 3, 64, 64)

    with model.inspect() as inspector:
        with torch.no_grad():
            _ = model(dummy)

    renderer = MermaidRenderer(show_legend=True, detail_level="none")
    mermaid = inspector.export(renderer, path="out/complex.mmd")
    print("\n" + mermaid)

    # Basic sanity: graph emitted and reused module expanded to unique nodes.
    assert "graph TD" in mermaid
    assert "shared_relu_0" in mermaid and "shared_relu_1" in mermaid and "shared_relu_2" in mermaid
    # Branch coverage: both branch1 and branch2 leaves should appear (named_modules indices).
    assert "branch1.0_0" in mermaid  # first conv in branch1
    assert "branch2_pool_0" in mermaid


class DeeperNet(nn.Module):
    """Deeper multi-branch network with shared layers reused across branches."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.shared_relu = nn.ReLU(inplace=True)
        self.shared_conv1x1 = nn.Conv2d(32, 32, kernel_size=1)

        # Three parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.branch2_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        res = self.block(x)
        res = self.shared_relu(res)  # reuse 1 on residual

        b1 = self.branch1(res)
        b1 = self.shared_conv1x1(self.shared_relu(b1))  # reuse relu + shared conv

        b2 = self.branch2_pool(res)
        b2 = self.shared_relu(b2)  # reuse 2
        b2 = self.branch2_conv(b2)

        b3 = self.branch3(res)
        b3 = self.shared_relu(b3)  # reuse 3

        merged = b1 + b2 + b3
        merged = self.shared_conv1x1(merged)  # shared conv reused second time
        out = self.tail(merged)
        return out.mean(dim=(2, 3))


def test_deeper_complex_mermaid_export() -> None:
    enable_dev_mode()

    model = DeeperNet().eval()
    dummy = torch.randn(1, 3, 64, 64)

    with model.inspect() as inspector:
        with torch.no_grad():
            _ = model(dummy)

    renderer = MermaidRenderer(show_legend=True, detail_level="none")
    mermaid = inspector.export(renderer, path="out/complex_deep.mmd")
    print("\n" + mermaid)

    # Basic sanity
    assert "graph TD" in mermaid
    # Shared ReLU is reused across residual and branches
    assert "shared_relu_0" in mermaid and "shared_relu_1" in mermaid and "shared_relu_2" in mermaid
    # Shared 1x1 conv used twice should emit unique nodes
    assert "shared_conv1x1_0" in mermaid and "shared_conv1x1_1" in mermaid
    # Branch coverage: expect leaves from three branches
    assert "branch1.0_0" in mermaid
    assert "branch2_pool_0" in mermaid
    assert "branch3.0_0" in mermaid
