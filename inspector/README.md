# Inspector (PyTorch Monkey-Patch Prototype)

## 概览
- 目标：在不修改 PyTorch 源码的前提下，为 `torch.nn.Module` 原生提供 `inspect()` 方法，完成模型拓扑提取、运行时观测（形状/统计/耗时/显存）、热力图可视化。
- 方法：通过运行时 monkey patch 将 `inspect` 挂到 `torch.nn.Module`，以上下文管理器收集前向数据并渲染 Mermaid 图。
- 模式：Facade（Inspector 入口）、Context Manager（RAII 清理 hook）、Observer（forward/forward_pre hook）、Strategy（MermaidRenderer）。

## 快速开始
```bash
# 安装依赖（建议 Python 3.11+）
uv sync  # 或 pip install -r requirements

uv run python -m pytest  # 运行全部测试
```

使用示例：
```python
from patch_utils import enable_dev_mode
from inspector.renderers import MermaidRenderer
import torch
from torchvision.models import resnet18

enable_dev_mode()
model = resnet18(weights=None).eval()

dummy = torch.randn(1, 3, 224, 224)
with model.inspect() as inspector:
	with torch.no_grad():
		_ = model(dummy)

renderer = MermaidRenderer(layout="LR", layout_engine="elk", collapse_level=2, show_legend=True)
mermaid = inspector.export(renderer, path="out/graph.mmd")
print(mermaid)
```

## 仓库结构
```
.
├── README.md                # 项目说明（本文件）
├── pyproject.toml           # 项目/依赖配置
├── uv.lock                  # 依赖锁文件
├── main.py                  # 可选入口（示例/预留）
├── src/
│   ├── inspector/
│   │   ├── __init__.py
│   │   ├── core.py          # Inspector 上下文管理与导出
│   │   ├── collector.py     # 运行时数据采集、hook 管理
│   │   ├── graph.py         # Graph/Node 数据结构
│   │   ├── renderers.py     # Renderer 抽象与 MermaidRenderer 实现
│   └── patch_utils.py       # enable_dev_mode: monkey patch torch.nn.Module.inspect
├── tests/
│   ├── conftest.py          # 测试路径设置
│   ├── test_demo.py         # ResNet18 集成示例
│   ├── test_complex_demo.py # 复杂分支网络集成测试
│   ├── test_transformer_demo.py # Transformer 编码器集成测试
│   └── test_inspector_unit.py # 单元/回归测试（复用模块、inplace、自环抑制等）
├── out/
│   └── graph.mmd            # 示例导出（可运行后生成）
└── .venv/                   # 虚拟环境（本地）
```

## 测试
```bash
uv run python -m pytest  # 运行全部测试
```

### 测试用例说明

#### 集成测试
- **[tests/test_demo.py](tests/test_demo.py)**: ResNet18 端到端测试。执行前向 + 反向（loss.backward），导出 Mermaid（含 legend），验证完整的采集/渲染链路。

- **[tests/test_complex_demo.py](tests/test_complex_demo.py)**: 复杂分支网络测试。测试 `ComplexNet` 模型，包含分支结构、共享 ReLU 模块的多次复用、以及 in-place 操作的正确追踪。验证边桥接、模块复用标记等高级特性。

- **[tests/test_transformer_demo.py](tests/test_transformer_demo.py)**: Transformer 编码器测试。测试 `TinyTransformer` 模型，包含多头注意力和前馈网络。验证深层嵌套结构的图生成，确保 LayerNorm、Linear 等组件正确显示。

#### 单元测试
[tests/test_inspector_unit.py](tests/test_inspector_unit.py) 包含 14 个单元测试，覆盖核心功能和边界情况：

**基础功能测试**：
- `test_graph_edges_inferred`: tensor id 推断顺序边，节点带调用后缀（fc1_0 -> relu_0 -> fc2_0）。
- `test_handles_removed_on_exit`: 上下文退出后 hook 句柄被清理，避免泄漏。
- `test_activation_and_gradient_stats_collected`: 前向+反向后，节点包含激活均值/方差与梯度均值/方差。

**渲染器测试**：
- `test_renderer_uses_br_and_hot_class`: 标签换行 `<br>` 正常，热点节点标记 hot，最热叶子填充为红色。
- `test_renderer_uses_leaf_only_hot`: 父容器不被标红，叶子热点才上色。
- `test_subgraph_hierarchy_emitted`: 顶层容器输出 subgraph，虚线边框样式。
- `test_no_nested_subgraph_for_basicblock`: 仅顶层容器生成 subgraph，内部 BasicBlock 不再嵌套盒子。
- `test_collapse_depth_filters_nested_nodes`: collapse_depth 参数正确过滤嵌套节点，仅显示指定深度。
- `test_container_edge_bridged_to_leaves`: 容器间边桥接到叶子节点，避免指向子图 ID。

**拓扑推断测试**：
- `test_reused_module_creates_unique_nodes`: 复用模块（同一 ReLU 调两次）生成带序号的独立节点，链路 fc_0 -> relu_0 -> relu_1。
- `test_inplace_relu_edges_do_not_bypass`: in-place ReLU 不被旁路，路径 fc1_0 -> relu_0 -> fc2_0，无直接 fc1_0 -> fc2_0。
- `test_implicit_edges_render_dashed`: 从模块容器直接到其内部叶子节点的隐式边渲染为虚线样式。

**边界情况测试**：
- `test_pattern_filters_modules`: pattern 过滤仅采集匹配模块（m1*），m2 被过滤。
- `test_no_tensor_objects_stored`: 导出的 graph 字典不包含 Tensor 实例，仅元数据。
