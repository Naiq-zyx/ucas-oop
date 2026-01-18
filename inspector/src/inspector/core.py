"""Inspector facade that manages graph collection lifecycle."""
from __future__ import annotations

import fnmatch
from typing import List, Optional
from abc import ABC, abstractmethod

import torch

from .collector import DataCollector
from .graph import Graph
from .renderers import Renderer


class InspectorInterface(ABC):
    @abstractmethod
    def __enter__(self) -> "InspectorInterface":
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    @abstractmethod
    def export(self, renderer: Renderer, path: Optional[str] = None) -> str:
        ...


class Inspector(InspectorInterface):
    """Context manager that collects runtime insights for a model.

    Usage:
        with Inspector(model) as inspector:
            _ = model(dummy_input)
            print(inspector.export(MermaidRenderer()))
    """

    def __init__(self, model: torch.nn.Module, pattern: str = "*") -> None:
        self.model = model
        self.pattern = pattern
        self.graph: Optional[Graph] = None
        self.collector: Optional[DataCollector] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "Inspector":
        self.graph = Graph()
        self.collector = DataCollector(self.graph)

        assert self.collector is not None  # for type checkers
        # Register a root hook on the model to capture user inputs as a virtual __input__ node.
        root_handle = self.model.register_forward_pre_hook(
            lambda mod, inputs: self.collector.register_root_inputs(inputs)
        )

        for name, module in self.model.named_modules():
            target_name = name or module.__class__.__name__
            if fnmatch.fnmatch(target_name, self.pattern):
                self.collector.register_module(target_name, module)

        self._handles = [root_handle, *self.collector.handles]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.collector:
            self.collector.remove_handles()
            self.collector.clear()
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def export(self, renderer: Renderer, path: Optional[str] = None) -> str:
        """Render the collected graph with a renderer and optionally write to disk."""

        if self.graph is None:
            raise RuntimeError("Inspector must be used as a context manager before exporting.")

        output = renderer.render(self.graph)

        if path:
            import os

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)

        return output
