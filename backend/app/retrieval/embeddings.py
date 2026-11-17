"""文本嵌入客户端：BGE-M3（远端）/ 本地哈希 n-gram（离线降级）。

离线模式并非随机向量：对字符 2/3-gram 做带符号哈希投影并 L2 归一化，
文本越相似、共享 n-gram 越多，余弦相似度越高——足以支撑混合检索、
RRF 融合与镜像缓存相似度判断在离线环境下产生"像真的"排序结果。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from backend.app.config import Settings, get_settings


class EmbeddingClient:
    """统一嵌入入口；上层（vector_store/hybrid/cache）只依赖本类。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dim = self.settings.embed_dim

    # ---------- 对外 ----------
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.settings.mock_mode:
            return [self._hash_embed(t) for t in texts]
        return self._embed_remote(texts)

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    # ---------- 离线实现 ----------
    def _hash_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        norm_text = re.sub(r"\s+", "", text.lower())
        grams = set()
        for n in (2, 3):
            for i in range(len(norm_text) - n + 1):
                grams.add(norm_text[i:i + n])
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0
            norm = 1.0
        return [v / norm for v in vec]

    # ---------- 远端实现 ----------
    def _embed_remote(self, texts: List[str]) -> List[List[float]]:
        import httpx

        url = self.settings.openai_base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {"model": self.settings.embed_model, "input": texts}
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]


_embedding_singleton: EmbeddingClient | None = None


def get_embedder() -> EmbeddingClient:
    """进程级嵌入单例。"""
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = EmbeddingClient()
    return _embedding_singleton
