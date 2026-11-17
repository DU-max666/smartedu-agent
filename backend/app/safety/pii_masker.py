"""未成年人 PII 脱敏器：规则 NER + 正则，占位符替换，生成后回填还原。

合规设计（对应"不留存、不训练"承诺）：
1. 占位符形如 [PII:姓名:3]，进入模型上下文的永远是脱敏文本；
2. 占位符→原值的映射表只保存在校内会话存储（内存/校内 Redis），
   绝不写入模型请求体、日志与外部缓存；
3. 脱敏失败即拒答：若文本命中疑似 PII 但无法可靠定位边界
   （ok=False），调用方必须拒绝把原文发给模型，走转人工话术。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 类别 -> 编译后的正则（捕获组为待脱敏内容）
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("手机号", re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")),
    ("身份证", re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")),
    ("邮箱", re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")),
    ("学号", re.compile(r"学号[：:\s]*([A-Za-z0-9\-]{4,20})")),
    ("姓名", re.compile(r"(?:学生|同学|老师|教师|家长|班主任)[：:\s]*([一-龥]{2,4})(?=[，。、；！？\s：:]|$)")),
    ("住址", re.compile(r"(?:住址|家庭地址|家住)[：:\s]*([^\s，。；]{4,40})")),
]

# 疑似但边界不可靠的模式：命中且无法用上面规则覆盖时 -> 拒答
_SUSPICIOUS = re.compile(r"(?<!\d)(\d{15,})(?!\d)")


@dataclass
class MaskResult:
    """脱敏结果。ok=False 表示存在不可靠 PII，调用方应拒答而非送模型。"""

    masked_text: str
    ok: bool = True
    reason: str = ""
    hits: List[Tuple[str, str]] = field(default_factory=list)  # (类别, 占位符)


class PiiMasker:
    """按会话维护映射表的脱敏器。"""

    def __init__(self) -> None:
        # 映射表仅存进程内（生产部署在校内 Redis，TTL 与会话一致）
        self._mappings: Dict[str, Dict[str, str]] = {}

    # ---------- 脱敏 ----------
    def mask(self, text: str, session_id: str) -> MaskResult:
        mapping = self._mappings.setdefault(session_id, {})
        masked = text
        hits: List[Tuple[str, str]] = []
        for category, pattern in _PATTERNS:
            for match in list(pattern.finditer(masked)):
                value = match.group(1)
                if value in mapping.values():
                    placeholder = next(k for k, v in mapping.items() if v == value)
                else:
                    idx = sum(1 for k in mapping if k.startswith(f"[PII:{category}:")) + 1
                    placeholder = f"[PII:{category}:{idx}]"
                    mapping[placeholder] = value
                masked = masked.replace(value, placeholder)
                hits.append((category, placeholder))
        # 兜底：仍有长数字串疑似证件号但未被规则捕获 -> 判定脱敏不可靠
        if _SUSPICIOUS.search(masked):
            leftover = [m.group(1) for m in _SUSPICIOUS.finditer(masked)]
            if leftover:
                return MaskResult(masked_text=text, ok=False,
                                  reason=f"疑似证件/卡号无法安全定位: {leftover[0][:4]}****",
                                  hits=hits)
        return MaskResult(masked_text=masked, ok=True, hits=hits)

    # ---------- 还原 ----------
    def restore(self, text: str, session_id: str) -> str:
        mapping = self._mappings.get(session_id, {})
        restored = text
        for placeholder, value in mapping.items():
            restored = restored.replace(placeholder, value)
        return restored

    def contains_pii(self, text: str) -> bool:
        return any(p.search(text) for _, p in _PATTERNS) or bool(_SUSPICIOUS.search(text))

    def drop_session(self, session_id: str) -> None:
        """会话结束即销毁映射表——落实"不留存"。"""
        self._mappings.pop(session_id, None)


_masker_singleton: Optional[PiiMasker] = None


def get_pii_masker() -> PiiMasker:
    global _masker_singleton
    if _masker_singleton is None:
        _masker_singleton = PiiMasker()
    return _masker_singleton
