"""Runtime data collection for the native inspector.

The collector registers forward hooks to gather lightweight metadata
about activations without retaining large tensors. Hooks emit:
- input/output shapes and dtypes
- simple activation statistics (mean/std) when inexpensive to compute
- execution time and approximate memory usage

The resulting information populates the shared Graph object, which will
later be rendered by a chosen backend.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Tuple
from abc import ABC, abstractmethod

import torch

from .graph import Graph, Node


class CollectorInterface(ABC):
    @abstractmethod
    def register_module(self, name: str, module: torch.nn.Module) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def remove_handles(self) -> None:
        ...


def _flatten_tensors(obj: Any) -> Iterable[torch.Tensor]:
    """Recursively yield tensors from nested structures."""

    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _flatten_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _flatten_tensors(item)


def _tensor_metadata(tensor: torch.Tensor) -> Dict[str, Any]:
    """Extract lightweight metadata for a tensor."""

    with torch.no_grad():
        meta: Dict[str, Any] = {
            "id": id(tensor),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "requires_grad": bool(tensor.requires_grad),
            "numel": tensor.numel(),
            "element_size": tensor.element_size(),
        }
        # Optional stats; computed lazily to avoid large overhead.
        try:
            meta["mean"] = float(tensor.detach().float().mean().item())
            meta["std"] = float(tensor.detach().float().std().item())
        except Exception:
            meta["mean"] = None
            meta["std"] = None
        return meta


def _memory_allocated_bytes() -> int:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return int(torch.cuda.memory_allocated(device))
    return 0


class DataCollector(CollectorInterface):
    """Manages PyTorch hooks and populates the computation graph."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._pre_info: Dict[str, Dict[str, Any]] = {}
        self._param_cache: Dict[str, int] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._alias_counter: int = 0
        self._call_counters: Dict[str, int] = {}
        self._last_node_for_module: Dict[str, str] = {}
        self._param_handles: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}
        self._root_registered: bool = False

    def register_module(self, name: str, module: torch.nn.Module) -> None:
        """Attach forward and forward-pre hooks to a module."""
        # Skip generic containers, but allow certain modules with children (e.g., MultiheadAttention).
        allow_with_children = (torch.nn.MultiheadAttention,)
        if any(module.children()) and not isinstance(module, allow_with_children):
            return

        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        self._param_cache[name] = param_count

        pre_handle = module.register_forward_pre_hook(
            lambda mod, inputs, *, _name=name: self._on_pre_hook(_name, mod, inputs)
        )
        post_handle = module.register_forward_hook(
            lambda mod, inputs, output, *, _name=name: self._on_post_hook(
                _name, mod, inputs, output
            )
        )
        param_handles: List[torch.utils.hooks.RemovableHandle] = []
        for p in module.parameters(recurse=False):
            if not p.requires_grad:
                continue
            handle = p.register_hook(
                lambda grad, *, _name=name: self._on_param_grad(_name, grad)
            )
            param_handles.append(handle)

        self._param_handles[name] = param_handles
        self.handles.extend([pre_handle, post_handle, *param_handles])

    def clear(self) -> None:
        self._pre_info.clear()
        self._last_node_for_module.clear()
        self._param_handles.clear()
        self._root_registered = False

    def remove_handles(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                # Removal failures are non-fatal; continue cleanup.
                pass
        self.handles.clear()

    def _on_pre_hook(self, name: str, module: torch.nn.Module, inputs: Tuple[Any, ...]) -> None:
        start_time = time.perf_counter()
        pre_mem = _memory_allocated_bytes()
        input_tensors = list(_flatten_tensors(inputs))
        input_meta = [_tensor_metadata(t) for t in input_tensors]
        call_idx = self._call_counters.get(name, 0)
        self._call_counters[name] = call_idx + 1
        unique_name = f"{name}_{call_idx}"
        self._pre_info[name] = {
            "start_time": start_time,
            "pre_mem": pre_mem,
            "inputs": input_meta,
            "unique_name": unique_name,
        }

    def _on_post_hook(
        self,
        name: str,
        module: torch.nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        end_time = time.perf_counter()
        post_mem = _memory_allocated_bytes()
        pre = self._pre_info.pop(name, {"start_time": end_time, "pre_mem": post_mem, "inputs": []})

        # If module returns a tuple (e.g., MultiheadAttention returns (out, attn)),
        # use the first tensor as the primary output for connectivity and stats.
        primary_output = output[0] if isinstance(output, tuple) else output
        output_tensors = list(_flatten_tensors(primary_output))
        output_meta = [_tensor_metadata(t) for t in output_tensors]

        exec_time_ms = (end_time - pre.get("start_time", end_time)) * 1000.0
        mem_delta = max(post_mem - pre.get("pre_mem", post_mem), 0)
        # Fallback to activation size when CUDA is unavailable.
        activation_bytes = sum(m["numel"] * m["element_size"] for m in output_meta)
        memory_bytes = mem_delta or activation_bytes

        input_ids = [m["id"] for m in pre.get("inputs", [])]
        # Register virtual input producers for user-provided tensors so first layer connects.
        self.register_root_inputs(pre.get("inputs", []))

        input_shapes = [m["shape"] for m in pre.get("inputs", [])]
        output_shapes = [m["shape"] for m in output_meta]
        input_dtypes = [m["dtype"] for m in pre.get("inputs", [])]
        output_dtypes = [m["dtype"] for m in output_meta]

        unique_name = pre.get("unique_name", name)

        # Aggregate activation stats for quick access.
        act_means = [m.get("mean") for m in output_meta if m.get("mean") is not None]
        act_stds = [m.get("std") for m in output_meta if m.get("std") is not None]
        act_mean = float(sum(act_means) / len(act_means)) if act_means else None
        act_std = float(sum(act_stds) / len(act_stds)) if act_stds else None

        node = Node(
            name=unique_name,
            module_type=module.__class__.__name__,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            parameter_count=self._param_cache.get(name, 0),
            memory_bytes=memory_bytes,
            exec_time_ms=exec_time_ms,
            act_mean=act_mean,
            act_std=act_std,
            stats={
                "input_stats": [{"mean": m.get("mean"), "std": m.get("std")} for m in pre.get("inputs", [])],
                "output_stats": [{"mean": m.get("mean"), "std": m.get("std")} for m in output_meta],
            },
        )

        input_id_set = set(input_ids)
        output_ids: List[int] = []
        alias_pairs: List[Tuple[int, int]] = []  # (original_id, synthetic_id)
        for m in output_meta:
            tid = m["id"]
            if tid in input_id_set:
                # In-place op reuses storage; create a synthetic id to keep DAG acyclic.
                self._alias_counter += 1
                synthetic_id = -self._alias_counter
                alias_pairs.append((tid, synthetic_id))
                tid = synthetic_id
                m["id"] = tid
            output_ids.append(tid)
        self.graph.add_edges_from_inputs(unique_name, input_ids)
        self.graph.register_producers(output_ids, unique_name)
        # Also map original tensor ids to this node to ensure downstream edges hit the op,
        # even when in-place reuse keeps the same storage id.
        for original_id, _synthetic in alias_pairs:
            self.graph.tensor_producers[original_id] = unique_name
        self.graph.add_node(node)
        # Track the last node name for this module to attach gradient stats later.
        self._last_node_for_module[name] = unique_name

    def register_root_inputs(self, inputs: Tuple[Any, ...]) -> None:
        if self._root_registered:
            return
        tensors = list(_flatten_tensors(inputs))
        if not tensors:
            return
        input_node = self.graph.nodes.get("__input__") or Node(name="__input__", module_type="Input")
        for t in tensors:
            tid = id(t)
            if tid in self.graph.tensor_producers:
                continue
            input_node.output_shapes.append(tuple(t.shape))
            input_node.output_dtypes.append(str(t.dtype))
            self.graph.tensor_producers[tid] = "__input__"
        self.graph.add_node(input_node)
        self._root_registered = True
    def _on_param_grad(self, name: str, grad: torch.Tensor) -> None:
        node_name = self._last_node_for_module.get(name)
        if node_name is None:
            return
        node = self.graph.nodes.get(node_name)
        if node is None:
            return
        try:
            g = grad.detach().float()
            node.grad_mean = float(g.mean().item())
            node.grad_std = float(g.std().item())
        except Exception:
            return
