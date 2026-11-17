"""题库工具：题目检索与组卷取样。

内置一份自创样例题库（离线可跑）；联网部署时可替换为学校题库 API，
run() 的返回结构保持不变，调用方零改动。
"""
from __future__ import annotations

from typing import Any, Dict, List

TOOL_SPEC: Dict[str, Any] = {
    "name": "query_question_bank",
    "description": "按学科/年级/知识点/难度从题库取题，用于组卷与批改时匹配评分要点。",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "学科"},
            "grade": {"type": "string", "description": "学段年级"},
            "topic": {"type": "string", "description": "知识点关键词，如 单位换算"},
            "difficulty": {"type": "integer", "minimum": 1, "maximum": 5,
                           "description": "难度 1-5，0 表示不限"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
        },
        "required": ["subject"],
        "additionalProperties": False,
    },
}

# 自创样例题目（内容为演示用途原创编写，非真实教材）
_SAMPLE_BANK: List[Dict[str, Any]] = [
    {"qid": "MATH-3A-001", "subject": "数学", "grade": "小学三年级", "topic": "单位换算",
     "type": "objective", "difficulty": 2, "stem": "3 千米 = （ ）米",
     "options": ["300", "3000", "30000"], "answer": "3000", "points": ["进率 1000 使用正确"]},
    {"qid": "MATH-3A-002", "subject": "数学", "grade": "小学三年级", "topic": "单位换算",
     "type": "subjective", "difficulty": 3,
     "stem": "比较 5 米 40 厘米 与 54 分米 的大小，并说明理由。",
     "answer": "相等", "points": ["统一单位为分米（要点1）", "5米=50分米、40厘米=4分米（要点2）",
                                  "得出 54分米=54分米 结论（要点3）"]},
    {"qid": "MATH-3A-003", "subject": "数学", "grade": "小学三年级", "topic": "周长",
     "type": "objective", "difficulty": 2, "stem": "边长 6 厘米的正方形周长是（ ）厘米。",
     "options": ["12", "24", "36"], "answer": "24", "points": ["周长公式正确"]},
    {"qid": "SCI-4B-001", "subject": "科学", "grade": "小学四年级", "topic": "水的三态",
     "type": "subjective", "difficulty": 2,
     "stem": "简述冬天窗户内侧出现水珠的原因。",
     "answer": "室内水蒸气遇冷液化成小水珠",
     "points": ["指出水蒸气来源（要点1）", "指出遇冷条件（要点2）", "正确使用液化概念（要点3）"]},
    {"qid": "ENG-7A-001", "subject": "英语", "grade": "初中七年级", "topic": "一般现在时",
     "type": "objective", "difficulty": 2,
     "stem": "He ____ (go) to school by bike every day.",
     "options": ["go", "goes", "going"], "answer": "goes", "points": ["第三人称单数变形"]},
    {"qid": "PHY-8A-001", "subject": "物理", "grade": "初中八年级", "topic": "密度",
     "type": "subjective", "difficulty": 3,
     "stem": "一个铁块质量 158g，密度 7.9g/cm³，求体积并写出公式变形过程。",
     "answer": "20 cm³", "points": ["公式 V=m/ρ（要点1）", "代入数值正确（要点2）",
                                     "单位 cm³（要点3）"]},
]


def run(subject: str, grade: str = "", topic: str = "",
        difficulty: int = 0, limit: int = 3) -> Dict[str, Any]:
    """题库过滤取样。"""
    picked = []
    for q in _SAMPLE_BANK:
        if q["subject"] != subject:
            continue
        if grade and q["grade"] != grade:
            continue
        if topic and topic not in q["topic"]:
            continue
        if difficulty and q["difficulty"] != difficulty:
            continue
        picked.append(dict(q))
        if len(picked) >= limit:
            break
    return {"questions": picked, "count": len(picked)}
