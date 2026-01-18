"""Unit tests for inspector correctness and robustness."""
from __future__ import annotations

import torch
import torch.nn as nn

from inspector.core import Inspector
from inspector.graph import Graph, Node
from inspector.renderers import MermaidRenderer


def _contains_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(_contains_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_tensor(v) for v in obj)
    return False


def test_graph_edges_inferred() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    model = TinyModel()
    x = torch.randn(1, 4)

    with Inspector(model) as inspector:
        _ = model(x)

    graph = inspector.graph
    assert graph is not None
    edges = set(graph.edges)
    # Expect edges between sequential layers inferred via tensor ids.
    assert ("fc1_0", "relu_0") in edges
    assert ("relu_0", "fc2_0") in edges
    # Nodes should be registered for each tracked module.
    assert {"fc1_0", "relu_0", "fc2_0"}.issubset(set(graph.nodes.keys()))


def test_handles_removed_on_exit() -> None:
    model = nn.Linear(2, 2)
    with Inspector(model) as inspector:
        _ = model(torch.randn(1, 2))
        handles_during = list(inspector.collector.handles)
    # Handles existed during collection and are cleared after exit.
    assert handles_during
    assert inspector.collector is not None
    assert inspector.collector.handles == []


def test_renderer_uses_br_and_hot_class() -> None:
    graph = Graph()
    node = Node(
        name="layer1",
        module_type="Linear",
        output_shapes=[(1, 2)],
        exec_time_ms=5.0,
        memory_bytes=1024,
        parameter_count=10,
    )
    graph.add_node(node)
    renderer = MermaidRenderer(memory_threshold=1, latency_threshold_ms=1)
    rendered = renderer.render(graph)
    assert "<br>" in rendered
    assert "&lt;br&gt;" not in rendered
    assert "class layer1 hot;" in rendered
    assert "style layer1 fill:#ff4d4f" in rendered  # max leaf turns red


def test_renderer_uses_leaf_only_hot() -> None:
    graph = Graph()
    parent = Node(name="seq", module_type="Sequential")
    child = Node(name="seq.conv", module_type="Conv2d", exec_time_ms=20.0)
    graph.add_node(parent)
    graph.add_node(child)
    renderer = MermaidRenderer(latency_threshold_ms=10.0, leaf_only_heatmap=True)
    rendered = renderer.render(graph)
    # parent should not be marked hot because it's a container
    assert "class seq hot;" not in rendered
    assert "class seq_conv hot;" in rendered


def test_activation_and_gradient_stats_collected() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = Tiny()
    x = torch.randn(2, 2)

    with Inspector(model) as inspector:
        y = model(x)
        loss = y.sum()
        loss.backward()

    graph = inspector.graph
    assert graph is not None
    node = graph.nodes.get("fc_0")
    assert node is not None
    # Activations and gradients should be captured
    assert node.act_mean is not None
    assert node.act_std is not None
    assert node.grad_mean is not None
    assert node.grad_std is not None


def test_subgraph_hierarchy_emitted() -> None:
    graph = Graph()
    parent = Node(name="block", module_type="Sequential")
    child = Node(name="block.fc", module_type="Linear")
    graph.add_node(parent)
    graph.add_node(child)
    renderer = MermaidRenderer()
    rendered = renderer.render(graph)
    assert "subgraph cluster_block [\"block\"]" in rendered
    assert "style cluster_block fill:none,stroke:#333,stroke-dasharray:5 5;" in rendered
    assert "block_fc[\"block.fc" in rendered


def test_no_nested_subgraph_for_basicblock() -> None:
    graph = Graph()
    graph.add_node(Node(name="layer1", module_type="Sequential"))
    graph.add_node(Node(name="layer1.0.conv1", module_type="Conv2d"))
    renderer = MermaidRenderer(collapse_nested=True)
    rendered = renderer.render(graph)
    # Only top-level layer1 should form a subgraph; BasicBlock stays flat.
    assert "subgraph cluster_layer1" in rendered
    assert "BasicBlock" not in rendered
    assert "subgraph cluster_layer1_0" not in rendered


def test_collapse_depth_filters_nested_nodes() -> None:
    graph = Graph()
    graph.add_node(Node(name="layer1", module_type="Sequential"))
    graph.add_node(Node(name="layer1.conv", module_type="Conv2d"))
    graph.add_node(Node(name="layer2", module_type="Sequential"))
    graph.add_node(Node(name="layer2.conv", module_type="Conv2d"))
    graph.edges.append(("layer1.conv", "layer2.conv"))

    renderer = MermaidRenderer(collapse_depth=1)
    rendered = renderer.render(graph)
    # Only top-level nodes rendered; nested conv nodes should be absent
    assert "layer1_conv" not in rendered
    assert "layer2_conv" not in rendered
    assert "layer1[\"layer1" in rendered
    assert "layer2[\"layer2" in rendered


def test_container_edge_bridged_to_leaves() -> None:
    graph = Graph()
    graph.add_node(Node(name="layer1", module_type="Sequential"))
    graph.add_node(Node(name="layer1.conv", module_type="Conv2d"))
    graph.add_node(Node(name="layer2", module_type="Sequential"))
    graph.add_node(Node(name="layer2.conv", module_type="Conv2d"))
    graph.edges.append(("layer1", "layer2"))
    renderer = MermaidRenderer(collapse_nested=True)
    rendered = renderer.render(graph)
    # Expect edge between concrete leaves, not subgraph ids.
    assert "layer1_conv --> layer2_conv" in rendered


def test_implicit_edges_render_dashed() -> None:
    graph = Graph()
    graph.add_node(Node(name="a", module_type="Linear"))
    graph.add_node(Node(name="b", module_type="Linear"))
    graph.implicit_edges.append(("a", "b"))
    renderer = MermaidRenderer()
    rendered = renderer.render(graph)
    assert "a --> b" in rendered
    assert "linkStyle 0 stroke-dasharray:5 5;" in rendered


def test_reused_module_creates_unique_nodes() -> None:
    class Reuse(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc(x)
            x = self.relu(x)
            x = self.relu(x)
            return x

    model = Reuse()
    x = torch.randn(1, 4)

    with Inspector(model) as inspector:
        _ = model(x)

    graph = inspector.graph
    assert graph is not None
    # Two distinct relu calls should produce two nodes with suffixes.
    relu_nodes = sorted([n for n in graph.nodes if n.startswith("relu_")])
    assert len(relu_nodes) == 2
    edges = set(graph.edges)
    # Ensure flow goes fc_0 -> relu_0 -> relu_1
    assert ("fc_0", relu_nodes[0]) in edges
    assert (relu_nodes[0], relu_nodes[1]) in edges


def test_inplace_relu_edges_do_not_bypass() -> None:
    class Inplace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = Inplace()
    x = torch.randn(1, 4)

    with Inspector(model) as inspector:
        _ = model(x)

    graph = inspector.graph
    assert graph is not None
    edges = set(graph.edges)
    # Ensure path goes fc1_0 -> relu_0 -> fc2_0, not fc1_0 -> fc2_0 directly.
    assert ("fc1_0", "relu_0") in edges
    assert ("relu_0", "fc2_0") in edges
    assert ("fc1_0", "fc2_0") not in edges


def test_pattern_filters_modules() -> None:
    class MixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m1 = nn.Linear(4, 4)
            self.m2 = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m2(self.m1(x))

    model = MixedModel()
    x = torch.randn(1, 4)

    with Inspector(model, pattern="m1*") as inspector:
        _ = model(x)

    graph = inspector.graph
    assert graph is not None
    assert any(n.startswith("m1") for n in graph.nodes)
    assert not any(n.startswith("m2") for n in graph.nodes)


def test_no_tensor_objects_stored() -> None:
    class Simple(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = Simple()
    x = torch.randn(2, 3)

    with Inspector(model) as inspector:
        _ = model(x)

    graph_dict = inspector.graph.to_dict()  # type: ignore[union-attr]
    assert not _contains_tensor(graph_dict)
