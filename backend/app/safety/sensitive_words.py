"""敏感词扫描：面向 K12 场景的最小内置词表 + 分级处置。

词表刻意保持克制（只覆盖自伤、暴力、赌博毒品、色情、霸凌等明确红线），
学校可按本地治理要求通过 add_words 扩展；命中后的处置由 risk_policy
决定（学生可见内容直接拦截，教师输入仅告警并记审计）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

_DEFAULT_WORDS: Dict[str, Set[str]] = {
    "自伤": {"自杀", "自残", "割腕", "不想活"},
    "暴力": {"打架", "斗殴", "持刀", "报复杀人"},
    "成瘾物": {"赌博", "吸毒", "毒品", "电子烟推销"},
    "色情": {"色情", "裸聊", "黄图"},
    "霸凌": {"霸凌", "校园欺凌", "孤立同学", "起侮辱外号"},
    "逃学": {"逃课攻略", "翻墙逃校", "替我签到"},
}


@dataclass
class SensitiveHit:
    """一次命中记录。"""

    word: str
    category: str
    severity: str  # block / warn


class SensitiveWordScanner:
    """多词表扫描器：内置分级词表，命中即返回最严重级别。"""

    def __init__(self, block_categories: Set[str] | None = None) -> None:
        self._words: Dict[str, tuple] = {}
        self.block_categories: Set[str] = block_categories or {"自伤", "色情", "成瘾物", "逃学"}
        self.add_words(_DEFAULT_WORDS, severity_by_category={
            "自伤": "block", "色情": "block", "成瘾物": "block", "逃学": "block",
            "暴力": "warn", "霸凌": "warn",
        })

    def add_words(self, groups: Dict[str, Set[str]],
                  severity_by_category: Dict[str, str] | None = None) -> None:
        severity_by_category = severity_by_category or {}
        for category, words in groups.items():
            severity = severity_by_category.get(
                category, "block" if category in self.block_categories else "warn")
            for w in words:
                self._words[w] = (category, severity)

    def scan(self, text: str) -> List[SensitiveHit]:
        hits: List[SensitiveHit] = []
        for word, (category, severity) in self._words.items():
            if word in text:
                hits.append(SensitiveHit(word=word, category=category, severity=severity))
        hits.sort(key=lambda h: h.severity != "block")
        return hits

    def should_block(self, hits: List[SensitiveHit]) -> bool:
        return any(h.severity == "block" for h in hits)
