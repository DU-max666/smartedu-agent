"""课标工具：按学段/学科/主题 strands 拉取课程标准要点。

优先走语料库检索（source_type=curriculum）；语料未初始化时回退到
内置的原创示例要点，保证备课链路在裸环境可演示。
"""
from __future__ import annotations

from typing import Any, Dict, List

TOOL_SPEC: Dict[str, Any] = {
    "name": "fetch_curriculum_points",
    "description": "获取指定学段、学科、主题下的课程标准要点条目，供备课大纲对齐教学目标。",
    "parameters": {
        "type": "object",
        "properties": {
            "grade": {"type": "string", "description": "学段年级，如 小学三年级"},
            "subject": {"type": "string", "description": "学科"},
            "strand": {"type": "string", "description": "课标主题/领域关键词，如 测量"},
        },
        "required": ["grade", "subject"],
        "additionalProperties": False,
    },
}

# 原创示例课标要点（演示用，非真实课标原文）
_FALLBACK_POINTS: Dict[str, List[str]] = {
    "数学|测量": [
        "结合实例认识长度单位，建立 1 千米、1 分米的长度观念",
        "掌握相邻长度单位间的进率，能进行简单换算",
        "能估测物体长度，并用合适单位表达结果",
    ],
    "数学|数与运算": [
        "理解四则运算的意义，经历算法多样化过程",
        "能进行整数四则运算并养成验算习惯",
    ],
    "科学|物质状态": [
        "观察并描述水的三态变化，能用温度解释变化条件",
        "关注生活中的物态变化现象并尝试解释",
    ],
    "物理|密度": [
        "理解密度是物质的一种属性，会用 V=m/ρ 进行简单计算",
        "经历测量固体密度的实验过程，规范记录数据",
    ],
}


def run(grade: str, subject: str, strand: str = "") -> Dict[str, Any]:
    """返回课标要点列表；语料命中优先，未命中回退内置示例。"""
    points: List[str] = []
    source = "builtin_fallback"
    try:
        from backend.app.tools.kb_tool import run as kb_run

        result = kb_run(query=f"{subject} {strand} 课程标准 要求",
                        source_type="curriculum", top_k=5)
        hits = result.get("hits", [])
        if hits:
            points = [h["text"] for h in hits]
            source = "corpus"
    except Exception:
        points = []

    if not points:
        for key, value in _FALLBACK_POINTS.items():
            k_subject, _, k_strand = key.partition("|")
            if k_subject == subject and (not strand or strand in k_strand):
                points.extend(value)
    return {"grade": grade, "subject": subject, "strand": strand,
            "points": points, "source": source}
