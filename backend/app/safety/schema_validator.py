"""轻量 JSON Schema 校验器（子集实现，零外部依赖）。

支持子集：type / properties / required / items / enum /
minimum / maximum / additionalProperties。
用途：对模型结构化输出（评分 JSON、大纲 JSON）做入库前校验，
校验失败触发网关纠错重试或降级转人工，杜绝脏结构污染下游。
"""
from __future__ import annotations

from typing import Any, Dict, List


def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """返回错误列表；空列表即通过。"""
    errors: List[str] = []
    expected_type = schema.get("type")
    if expected_type and not _type_ok(instance, expected_type):
        return [f"{path}: 期望类型 {expected_type}，实际 {type(instance).__name__}"]

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: 值 {instance!r} 不在枚举 {schema['enum']} 中")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} 小于下限 {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} 大于上限 {schema['maximum']}")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: 缺少必填字段 {key}")
        props = schema.get("properties", {})
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: 出现未声明字段 {key}")

    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


def is_valid(instance: Any, schema: Dict[str, Any]) -> bool:
    return not validate(instance, schema)


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def _type_ok(value: Any, expected: str) -> bool:
    checker = _TYPE_CHECKS.get(expected)
    return checker(value) if checker else True
