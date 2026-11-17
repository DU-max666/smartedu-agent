"""Badcase 归因脚本：把评测失败案例自动归入四类根因并给出治理动作。

归因分类（与 docs/data_governance.md 的闭环机制一致）：
- retrieval_miss  检索缺失：证据未命中 -> 补语料/调切片
- generation_hallucination 生成幻觉：证据命中但引用/一致性失败 -> 收紧提示词
- prompt_ambiguity 提示歧义：路由意图判错 -> 补意图样本
- data_quality    数据质量：脱敏/敏感词等安全断言失败 -> 修数据管道
用法：
    python eval/run_eval.py --out eval/results.jsonl
    python eval/badcase_analysis.py eval/results.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_CATEGORY = {
    "router": "prompt_ambiguity",
    "qa": "retrieval_miss",
    "grading_objective": "data_quality",
    "grading_subjective": "data_quality",
    "profile": "data_quality",
    "safety_pii": "data_quality",
    "safety_sensitive": "data_quality",
}

_ACTION = {
    "retrieval_miss": "检查语料覆盖与切片粒度；必要时重跑 build_corpus 并调低子块长度",
    "generation_hallucination": "收紧生成提示词的引用纪律；核对 citation_checker 坐标是否入库",
    "prompt_ambiguity": "补充该意图的样本词/微调数据；复核派发闸值",
    "data_quality": "回查入库清洗规则与脱敏正则；必要时下线坏样本（--retire）",
}


def classify_failure(result: Dict[str, Any]) -> str:
    """依据失败细节做规则归因（可逐步替换为模型归因）。"""
    kind = result["type"]
    detail = result.get("detail", {})
    if kind == "qa":
        if not detail.get("hit"):
            return "retrieval_miss"
        return "generation_hallucination"
    return _CATEGORY.get(kind, "data_quality")


def analyze(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    bad = [r for r in results if not r.get("passed")]
    buckets: Counter = Counter()
    items: Dict[str, List[str]] = {}
    for r in bad:
        cat = classify_failure(r)
        buckets[cat] += 1
        items.setdefault(cat, []).append(r["case_id"])
    return {"total": len(results), "bad": len(bad),
            "distribution": dict(buckets), "case_ids": items}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = [json.loads(line) for line in
            Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = analyze(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for cat in report["distribution"]:
        print(f"[{cat}] 处置建议：{_ACTION[cat]}")
    if report["bad"] == 0:
        print("本轮无 badcase，归因闭环未触发。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
