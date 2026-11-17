"""会话记忆：最近 N 轮原文 + 更早轮次的滚动摘要。

滚动摘要策略（区别于"全量历史塞 prompt"的常见模板）：
- 保留最近 N 轮逐字原文（保证指代消解与追问连贯）；
- 溢出部分折叠进一条摘要：mock 模式下用确定性截断拼接，
  联网模式调用网关生成一句话增量摘要；
- 摘要与原文都只存内存/校内 Redis，随会话销毁，不落长期库。
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List

from backend.app.config import get_settings


class SessionMemory:
    """按 session_id 隔离的多轮记忆。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._turns: Dict[str, Deque[Dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=get_settings().recent_turns * 4))
        self._summaries: Dict[str, str] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            turns = self._turns[session_id]
            turns.append({"role": role, "content": content})
            recent_n = get_settings().recent_turns
            if len(turns) > recent_n:
                overflow = list(turns)[:-recent_n]
                for _ in range(len(overflow)):
                    turns.popleft()
                self._fold_summary(session_id, overflow)

    def recent(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    def summary(self, session_id: str) -> str:
        with self._lock:
            return self._summaries.get(session_id, "")

    def context_block(self, session_id: str) -> str:
        """拼装为提示词可用的历史块。"""
        with self._lock:
            lines = []
            summary = self._summaries.get(session_id, "")
            if summary:
                lines.append(f"早前对话摘要：{summary}")
            for turn in self._turns.get(session_id, []):
                who = "学生/用户" if turn["role"] == "user" else "助手"
                lines.append(f"{who}：{turn['content']}")
            return "\n".join(lines)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)
            self._summaries.pop(session_id, None)

    # ---------- 内部 ----------
    def _fold_summary(self, session_id: str, overflow: List[Dict[str, str]]) -> None:
        old_summary = self._summaries.get(session_id, "")
        if get_settings().mock_mode:
            # 确定性折叠：旧摘要 + 溢出轮次首片段，控制在 200 字内
            pieces = [old_summary] if old_summary else []
            pieces += [f"{t['role']}:{t['content'][:24]}" for t in overflow]
            self._summaries[session_id] = "；".join(pieces)[:200]
            return
        from backend.app.gateway.model_gateway import get_gateway

        lines = "\n".join(f"{t['role']}: {t['content']}" for t in overflow)
        prompt = (f"已有摘要：{old_summary or '（无）'}\n新增对话：\n{lines}\n"
                  "请合并为不超过 80 字的中文对话摘要，只输出摘要本身。")
        result = get_gateway().chat(
            [{"role": "system", "content": "你负责压缩对话历史。"},
             {"role": "user", "content": prompt}], scene="qa", max_tokens=160)
        self._summaries[session_id] = result.text.strip()[:200]


_memory_singleton: SessionMemory | None = None


def get_session_memory() -> SessionMemory:
    global _memory_singleton
    if _memory_singleton is None:
        _memory_singleton = SessionMemory()
    return _memory_singleton
