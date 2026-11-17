"""BM25 稀疏召回：自实现（不依赖第三方 rank_bm25）。

针对中文场景做了两处适配：
1. 分词采用"字符 bigram + 连续英文/数字整词"的轻量策略，
   避免引入外部分词器依赖，同时兼顾中英混合题干；
2. 语料索引支持增删（invalidate），配合数据治理管道做坏样本下线。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def tokenize(text: str) -> List[str]:
    """中英混合轻量分词：英文数字整词 + 中文单字再拼 bigram。"""
    raw = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in raw if not re.match(r"^[一-鿿]$", t)]
    han = [t for t in raw if re.match(r"^[一-鿿]$", t)]
    bigrams = [a + b for a, b in zip(han, han[1:])]
    return tokens + han + bigrams


class BM25Index:
    """Okapi BM25 内存索引。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_texts: Dict[str, str] = {}
        self._tf: Dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._avg_len = 0.0
        self._n_docs = 0

    # ---------- 索引维护 ----------
    def add_documents(self, docs: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in docs:
            if doc_id in self.doc_texts:
                self.invalidate(doc_id)
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._tf[doc_id] = tf
            for term in tf:
                self._df[term] += 1
        self._rebuild_stats()

    def invalidate(self, doc_id: str) -> None:
        """删除一篇文档（坏样本下线）。"""
        if doc_id not in self.doc_texts:
            return
        for term in self._tf[doc_id]:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        del self._tf[doc_id]
        del self.doc_texts[doc_id]
        self.doc_ids.remove(doc_id)
        self._rebuild_stats()

    def _rebuild_stats(self) -> None:
        self._n_docs = len(self.doc_ids)
        total_len = sum(sum(tf.values()) for tf in self._tf.values())
        self._avg_len = total_len / self._n_docs if self._n_docs else 0.0

    # ---------- 查询 ----------
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._n_docs == 0:
            return []
        scores: Dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in self._tf.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                doc_len = sum(tf.values())
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_len, 1.0))
                scores[doc_id] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
