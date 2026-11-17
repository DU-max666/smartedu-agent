"""工具子包：Function Calling 工具注册表。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from backend.app.tools import academic_affairs, curriculum_tool, kb_tool, question_bank

# 名称 -> (JSON Schema 规格, 可调用实现)
_REGISTRY: Dict[str, tuple] = {}


def _register(module: Any) -> None:
    _REGISTRY[module.TOOL_SPEC["name"]] = (module.TOOL_SPEC, module.run)


for _mod in (kb_tool, question_bank, curriculum_tool, academic_affairs):
    _register(_mod)


def get_tool_specs() -> List[Dict[str, Any]]:
    """OpenAI tools 参数格式。"""
    return [{"type": "function", "function": spec} for spec, _ in _REGISTRY.values()]


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """统一工具调用入口；未知工具名快速失败。"""
    if name not in _REGISTRY:
        return {"error": f"未注册的工具: {name}"}
    _, fn = _REGISTRY[name]
    return fn(**arguments)
