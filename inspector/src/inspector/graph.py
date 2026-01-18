"""Computation graph utilities for the native inspector.

This module defines light-weight data structures to describe a model
execution graph without storing tensors themselves. Nodes are enriched
with execution metadata (shapes, dtypes, timing, and memory footprint)
so renderers can highlight hotspots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Optional
from abc import ABC, abstractmethod


class NodeInterface(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...


class GraphInterface(ABC):
    @abstractmethod
    def add_node(self, node: "NodeInterface") -> None:
        ...

    @abstractmethod
    def register_producers(self, tensor_ids: Iterable[int], node_name: str) -> None:
        ...

    @abstractmethod
    def add_edges_from_inputs(self, node_name: str, input_tensor_ids: Iterable[int]) -> None:
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...


@dataclass
class Node(NodeInterface):
    """Represents a single module invocation in the graph."""

    name: str
    module_type: str
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    output_dtypes: List[str] = field(default_factory=list)
    parameter_count: int = 0
    memory_bytes: int = 0
    exec_time_ms: float = 0.0
    act_mean: float | None = None
    act_std: float | None = None
    grad_mean: float | None = None
    grad_std: float | None = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dict for downstream consumers."""

        return {
            "name": self.name,
            "module_type": self.module_type,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
            "parameter_count": self.parameter_count,
            "memory_bytes": self.memory_bytes,
            "exec_time_ms": self.exec_time_ms,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
            "grad_mean": self.grad_mean,
            "grad_std": self.grad_std,
            "stats": self.stats,
        }


class Graph(GraphInterface):
    """Minimal directed graph inferred from tensor flow."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str]] = []
        self.implicit_edges: List[Tuple[str, str]] = []
        self._last_node: Optional[str] = None
        # Maps id(tensor) to the node name that produced it.
        self.tensor_producers: Dict[int, str] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        self._last_node = node.name

    def register_producers(self, tensor_ids: Iterable[int], node_name: str) -> None:
        """Mark tensors as produced by a node using their python ids."""

        for tid in tensor_ids:
            self.tensor_producers[tid] = node_name

    def add_edges_from_inputs(self, node_name: str, input_tensor_ids: Iterable[int]) -> None:
        """Infer edges by matching input tensor ids to prior producers."""

        added = False
        missing = False
        for tid in input_tensor_ids:
            producer = self.tensor_producers.get(tid)
            if producer is None:
                missing = True
                continue
            edge = (producer, node_name)
            if edge[0] == edge[1]:
                # Suppress self-loops to keep DAG valid (e.g., in-place ops).
                continue
            if edge not in self.edges:
                self.edges.append(edge)
            added = True

        # If any inputs lacked a producer (e.g., residual add not seen), bridge from last node as implicit.
        if missing and self._last_node and self._last_node != node_name:
            edge = (self._last_node, node_name)
            if edge not in self.edges:
                self.edges.append(edge)
            if edge not in self.implicit_edges:
                self.implicit_edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        """Export the graph into a serializable structure."""

        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": list(self.edges),
            "implicit_edges": list(self.implicit_edges),
        }
