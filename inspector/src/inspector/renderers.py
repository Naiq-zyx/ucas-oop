"""Renderer implementations for inspector outputs."""
from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Set

from .graph import Graph, Node


class Renderer(ABC):
    """Abstract base class for graph renderers."""

    @abstractmethod
    def render(self, graph: Graph) -> str:
        """Render a computation graph into a textual representation."""


class MermaidRenderer(Renderer):
    """Render the computation graph as Mermaid flowchart code.

    Features:
    - Top-down layout (Mermaid TD).
    - Optional collapse of nested subgraphs; only top-level containers get clusters.
    - Leaf-focused heatmap coloring to reduce container bias.
    - Compact label formatting to reduce clutter.
    """

    def __init__(
        self,
        memory_threshold: int = 50 * 1024 * 1024,  # 50MB
        latency_threshold_ms: float = 10.0,
        max_depth: Optional[int] = None,
        collapse_depth: Optional[int] = None,
        leaf_only_heatmap: bool = True,
        collapse_nested: bool = True,
        detail_level: str = "full",  # "full", "compact", or "none"
        show_legend: bool = False,
        heatmap_mode: str = "runtime",  # "runtime" or "gradient"
    ) -> None:
        self.memory_threshold = memory_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.max_depth = max_depth
        self.collapse_depth = collapse_depth
        self.leaf_only_heatmap = leaf_only_heatmap
        self.collapse_nested = collapse_nested
        self.detail_level = detail_level
        self.show_legend = show_legend
        self.heatmap_mode = heatmap_mode

    def render(self, graph: Graph) -> str:
        lines: List[str] = ["graph TD"]

        def sanitize(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", name)

        depth_limit = self.collapse_depth if self.collapse_depth is not None else self.max_depth
        included = self._filter_by_depth(graph.nodes.keys(), depth_limit)
        children = self._build_children(included)
        leaves = {name for name in included if not children.get(name)}

        # Group nodes by top-level container for optional subgraph clustering.
        groups: Dict[str, List[str]] = {}
        for name in sorted(included):
            top = name.split(".")[0] if self.collapse_nested else name
            groups.setdefault(top, []).append(name)

        # Pre-compute scaling for heatmap using leaves only.
        leaf_times = [graph.nodes[n].exec_time_ms for n in leaves]
        leaf_mems = [graph.nodes[n].memory_bytes for n in leaves]
        max_time = max(leaf_times) if leaf_times else 0.0
        max_mem = max(leaf_mems) if leaf_mems else 0

        hot_nodes: List[str] = []
        node_styles: List[str] = []

        def render_node(node_name: str, indent: int) -> None:
            node = graph.nodes[node_name]
            node_id = sanitize(node_name)
            label = self._build_label(node)
            is_leaf = node_name in leaves
            if self._is_hot(node) and (is_leaf or not self.leaf_only_heatmap):
                hot_nodes.append(node_id)
            lines.append(f"{'    '*indent}{node_id}[\"{label}\"]")
            # Apply per-node heat color based on normalized leaf scores.
            score = 0.0
            if is_leaf:
                if self.heatmap_mode == "gradient" and node.grad_mean is not None:
                    # Placeholder: map gradient magnitude to score; small -> near 0, large -> near 1.
                    score = min(max(abs(node.grad_mean), 0.0), 1.0)
                else:
                    score = self._heat_score(node, max_time=max_time, max_mem=max_mem)
            color = self._color_for_score(score)
            node_styles.append(f"    style {node_id} fill:{color},stroke:#999,stroke-width:1px;")
        def render_group(group_name: str, members: List[str]) -> None:
            group_id = sanitize(group_name)
            cluster_id = f"cluster_{group_id}"  # avoid id collision with nodes
            if self.collapse_nested and len(members) > 1:
                lines.append(f"    subgraph {cluster_id} [\"{html.escape(group_name)}\"]")
                # Subtle grouping style: no fill, dashed stroke.
                lines.append(f"    style {cluster_id} fill:none,stroke:#333,stroke-dasharray:5 5;")
                for member in members:
                    render_node(member, 2)
                lines.append("    end")
            else:
                for member in members:
                    render_node(member, 1)

        for group_name, members in groups.items():
            render_group(group_name, members)

        # Render edges between actual tensor-processing nodes. If an edge connects
        # containers, bridge it to the deepest leaf of src and the first leaf of dst.
        rendered_edges = set()
        dashed_edge_indexes: List[int] = []
        edge_counter = 0

        def representative_leaf(name: str, prefer_first: bool) -> Optional[str]:
            if name in leaves:
                return name
            # Walk down the hierarchy to find a leaf.
            queue = children.get(name, [])
            if not queue:
                return None
            idx = 0 if prefer_first else -1
            child = queue[idx]
            while child in children and children[child]:
                queue = children[child]
                child = queue[0 if prefer_first else -1]
            return child if child in included else None

        def emit_edge(src: str, dst: str, dashed: bool = False) -> None:
            edge = (sanitize(src), sanitize(dst))
            if edge in rendered_edges:
                return
            rendered_edges.add(edge)
            lines.append(f"    {edge[0]} --> {edge[1]}")
            nonlocal edge_counter
            if dashed:
                dashed_edge_indexes.append(edge_counter)
            edge_counter += 1

        for src, dst in graph.edges:
            if src not in included or dst not in included:
                continue
            src_leaf = representative_leaf(src, prefer_first=False)
            dst_leaf = representative_leaf(dst, prefer_first=True)
            if src_leaf is None or dst_leaf is None:
                continue
            emit_edge(src_leaf, dst_leaf, dashed=False)

        # Implicit edges (e.g., residual adds) drawn as dashed.
        for src, dst in getattr(graph, "implicit_edges", []):
            if src not in included or dst not in included:
                continue
            src_leaf = representative_leaf(src, prefer_first=False)
            dst_leaf = representative_leaf(dst, prefer_first=True)
            if src_leaf is None or dst_leaf is None:
                continue
            emit_edge(src_leaf, dst_leaf, dashed=True)

        # Style definitions and per-node styles.
        lines.append("    classDef default fill:#f5f5f5,stroke:#999,stroke-width:1px;")
        lines.append("    classDef hot fill:#ffe2e2,stroke:#ff4d4f,stroke-width:2px;")
        lines.append("    linkStyle default interpolate basis;")
        lines.extend(node_styles)
        for idx in dashed_edge_indexes:
            lines.append(f"    linkStyle {idx} stroke-dasharray:5 5;")
        for node_id in hot_nodes:
            lines.append(f"    class {node_id} hot;")

        if self.show_legend:
            lines.extend(self._legend_block())

        return "\n".join(lines)

    def _filter_by_depth(self, names: Iterable[str], depth: Optional[int]) -> Set[str]:
        if depth is None:
            return set(names)
        return {n for n in names if n.count(".") + 1 <= depth}

    def _build_children(self, names: Set[str]) -> Dict[str, List[str]]:
        children: Dict[str, List[str]] = {}
        for name in sorted(names):
            parent = self._parent_of(name)
            if parent is not None and parent in names:
                children.setdefault(parent, []).append(name)
        return children

    def _parent_of(self, name: str) -> Optional[str]:
        parts = name.split(".")
        if len(parts) <= 1:
            return None
        return ".".join(parts[:-1])

    def _is_hot(self, node: Node) -> bool:
        return (
            node.memory_bytes >= self.memory_threshold
            or node.exec_time_ms >= self.latency_threshold_ms
        )

    def _heat_score(self, node: Node, *, max_time: float, max_mem: int) -> float:
        time_score = node.exec_time_ms / max_time if max_time > 0 else 0.0
        mem_score = node.memory_bytes / max_mem if max_mem > 0 else 0.0
        return max(time_score, mem_score)

    def _build_label(self, node: Node) -> str:
        parts: List[str] = [node.name, node.module_type]
        if self.detail_level == "full":
            if node.output_shapes:
                parts.append(f"out: {node.output_shapes}")
            if node.exec_time_ms:
                parts.append(f"Time={self._format_time(node.exec_time_ms)}")
            if node.memory_bytes:
                parts.append(f"Mem={self._format_memory(node.memory_bytes)}")
            if node.parameter_count:
                parts.append(f"params={self._format_params(node.parameter_count)}")
            if node.act_mean is not None or node.act_std is not None:
                parts.append(self._format_act(node))
            if node.grad_mean is not None or node.grad_std is not None:
                parts.append(self._format_grad(node))
        else:
            if self.detail_level == "compact":
                if node.exec_time_ms:
                    parts.append(f"t={self._format_time(node.exec_time_ms)}")
                if node.memory_bytes:
                    parts.append(f"m={self._format_memory(node.memory_bytes)}")
                if node.grad_mean is not None:
                    parts.append(self._format_grad(node))
            else:  # "none"
                parts = [node.name]
        escaped = [html.escape(p) for p in parts]
        return "<br>".join(escaped)

    def _format_params(self, n: int) -> str:
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}k"
        return str(n)

    def _format_memory(self, b: int) -> str:
        if b >= 1024 * 1024:
            return f"{b/(1024*1024):.2f}MB"
        if b >= 1024:
            return f"{b/1024:.1f}KB"
        return f"{b}B"

    def _format_time(self, ms: float) -> str:
        return f"{ms:.3f}ms" if ms >= 1.0 else f"{ms*1000:.1f}µs"

    def _format_act(self, node: Node) -> str:
        mu = f"{node.act_mean:.3f}" if node.act_mean is not None else "?"
        sigma = f"{node.act_std:.3f}" if node.act_std is not None else "?"
        return f"act: μ={mu}, σ={sigma}"

    def _format_grad(self, node: Node) -> str:
        mu = f"{node.grad_mean:.2e}" if node.grad_mean is not None else "?"
        sigma = f"{node.grad_std:.2e}" if node.grad_std is not None else "?"
        return f"grad: μ={mu}, σ={sigma}"

    def _color_for_score(self, score: float) -> str:
        """Interpolate between light fill and red; score in [0,1]."""

        score = max(0.0, min(1.0, score))
        start = (255, 245, 245)  # light
        end = (255, 77, 79)      # red
        r = int(start[0] + (end[0] - start[0]) * score)
        g = int(start[1] + (end[1] - start[1]) * score)
        b = int(start[2] + (end[2] - start[2]) * score)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _legend_block(self) -> List[str]:
        """Render a small legend explaining heatmap coloring."""

        hot = self._color_for_score(1.0)
        mid = self._color_for_score(0.5)
        cool = self._color_for_score(0.1)
        lines = ["    subgraph cluster_legend [\"Legend\"]", "    style cluster_legend fill:none,stroke:#333,stroke-dasharray:5 5;"]
        lines.append(f"    legend_hot[\"Highest time/mem\"]")
        lines.append(f"    legend_mid[\"Medium\"]")
        lines.append(f"    legend_cool[\"Low\"]")
        lines.append("    end")
        lines.append(f"    style legend_hot fill:{hot},stroke:#999,stroke-width:1px;")
        lines.append(f"    style legend_mid fill:{mid},stroke:#999,stroke-width:1px;")
        lines.append(f"    style legend_cool fill:{cool},stroke:#999,stroke-width:1px;")
        return lines
