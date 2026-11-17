"""重排器：BGE-Reranker（远端 cross-encoder）/ 词面覆盖度（离线降级）。

离线降级实现不是简单打分：用"查询词对候选文档的加权覆盖率"近似
cross-encoder 行为——稀有词权重高（IDF），并对长度做饱和惩罚，
保证降级路径下排序仍有区分度。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

from backend.app.config import Settings, get_settings
from backend.app.retrieval.bm25 import tokenize


class Reranker:
    """统一重排入口。score 越高越相关。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """candidates: [(item_id, text)] -> [(item_id, score)] 降序。"""
        if not candidates:
            return []
        if self.settings.mock_mode:
            return self._lexical_score(query, candidates)
        return self._remote_rerank(query, candidates)

    # ---------- 离线实现 ----------
    def _lexical_score(self, query: str,
                       candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        q_tokens = Counter(tokenize(query))
        doc_token_freq: Counter = Counter()
        for _, text in candidates:
            doc_token_freq.update(set(tokenize(text)))
        n_docs = max(len(candidates), 1)

        def idf(term: str) -> float:
            return math.log(1.0 + (n_docs - doc_token_freq.get(term, 0) + 0.5)
                            / (doc_token_freq.get(term, 0) + 0.5))

        scored: List[Tuple[str, float]] = []
        for item_id, text in candidates:
            doc_tokens = set(tokenize(text))
            cover = sum(idf(t) * (1 if t in doc_tokens else 0) for t in q_tokens)
            total = sum(idf(t) for t in q_tokens) or 1.0
            length_penalty = 1.0 / (1.0 + max(0, len(text) - 400) / 800.0)
            scored.append((item_id, cover / total * length_penalty))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    # ---------- 远端实现 ----------
    def _remote_rerank(self, query: str,
                       candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        import httpx

        url = self.settings.openai_base_url.rstrip("/") + "/rerank"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": [text for _, text in candidates],
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            results = resp.json()["results"]
        by_index = {r["index"]: float(r["relevance_score"]) for r in results}
        return sorted(
            [(item_id, by_index.get(i, 0.0)) for i, (item_id, _) in enumerate(candidates)],
            key=lambda kv: kv[1], reverse=True,
        )
