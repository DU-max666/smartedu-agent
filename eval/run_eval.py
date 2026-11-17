"""离线评测脚本：对 eval_set.jsonl 跑五类断言并输出指标表。

用法：
    python eval/run_eval.py                       # 全量评测（离线 mock 模式）
    python eval/run_eval.py --out results.jsonl   # 同时导出逐条结果供 badcase 归因

指标口径（与 README 评测表一致）：
- 路由准确率 / 答疑证据命中率 / 客观题判分准确率 /
  主观题一致率（与期望分差 ≤ 20% 满分视为一致）/ 掉队预警召回 / 安全拦截率。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SKIP_DB", "1")

from backend.app.data.pipeline import CorpusPipeline  # noqa: E402
from backend.app.graphs.grading import build_grading_graph  # noqa: E402
from backend.app.graphs.profile import build_profile_graph  # noqa: E402
from backend.app.graphs.qa import build_qa_graph  # noqa: E402
from backend.app.graphs.router_graph import get_router_graph  # noqa: E402
from backend.app.safety.pii_masker import get_pii_masker  # noqa: E402
from backend.app.safety.sensitive_words import SensitiveWordScanner  # noqa: E402


def load_cases(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Evaluator:
    """分类型断言执行器。"""

    def __init__(self) -> None:
        CorpusPipeline().ingest_dir(str(ROOT / "backend" / "app" / "data" / "seed_corpus"))
        self.router = get_router_graph()
        self.qa = build_qa_graph()
        self.grading = build_grading_graph()
        self.profile = build_profile_graph()
        self.scanner = SensitiveWordScanner()
        self.results: List[Dict[str, Any]] = []

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        kind = case["type"]
        handler = {
            "router": self._router, "qa": self._qa,
            "grading_objective": self._grading, "grading_subjective": self._grading,
            "profile": self._profile, "safety_pii": self._pii,
            "safety_sensitive": self._sensitive,
        }[kind]
        passed, detail = handler(case)
        out = {"case_id": case["case_id"], "type": kind, "passed": passed, "detail": detail}
        self.results.append(out)
        return out

    # ---------- 各类断言 ----------
    def _router(self, case: Dict[str, Any]):
        final = self.router.run({"session_id": f"eval-{case['case_id']}",
                                 "user_role": case["role"], "utterance": case["utterance"]})
        intent = final.get("intent", "")
        expect = case["expect_intent"]
        if expect == "clarify":
            ok = bool(final.get("clarify_question")) and not final.get("dispatch_target")
        elif expect == "clarify_or_blocked":
            ok = final.get("dispatch_target") != "lesson"
        else:
            ok = intent == expect and final.get("dispatch_target") == expect
        return ok, {"intent": intent, "conf": final.get("intent_confidence"),
                    "target": final.get("dispatch_target")}

    def _qa(self, case: Dict[str, Any]):
        final = self.qa.invoke({"question": case["question"], "grade": case.get("grade", ""),
                                "subject": case.get("subject", ""),
                                "session_id": f"eval-{case['case_id']}"})
        if case.get("expect_refuse"):
            return bool(final.get("refuse_reason")), {"refused": True}
        evidence = final.get("evidence", [])
        hit = any(e.get("doc_id") == case["expect_doc"] for e in evidence)
        # 引用断言：答案须引用任一命中证据（多文档同题时以实际最优证据为准）
        cited = any(e.get("doc_id") and e["doc_id"] in (final.get("answer") or "")
                    for e in evidence)
        return hit and cited and bool(final.get("citation_valid")), \
            {"hit": hit, "cited": cited, "valid": final.get("citation_valid")}

    def _grading(self, case: Dict[str, Any]):
        final = self.grading.invoke({"items": case["items"], "student_ref": "S-eval",
                                     "task_id": case["case_id"]})
        rows = final.get("objective_results", []) + final.get("subjective_results", [])
        by_qid = {r["qid"]: r for r in rows}
        ok = True
        for exp in case.get("expect", []):
            got = by_qid.get(exp["qid"], {})
            if "correct" in exp and bool(got.get("correct")) != exp["correct"]:
                ok = False
            if "score" in exp and abs(float(got.get("score", -1)) - exp["score"]) > 0.01:
                ok = False
        if "expect_min_score" in case:
            got = list(by_qid.values())[0] if by_qid else {}
            ok = float(got.get("score", 0)) >= case["expect_min_score"]
        if "expect_review" in case:
            need_review = bool(final.get("low_confidence_items"))
            ok = need_review == case["expect_review"]
        return ok, {"results": rows, "review": final.get("review_status")}

    def _profile(self, case: Dict[str, Any]):
        final = self.profile.invoke({"student_ref": case["student_ref"],
                                     "knowledge_tree": case.get("knowledge_tree", {})})
        got = set(final.get("lagging_nodes", []))
        return got == set(case.get("expect_lagging", [])), {"lagging": sorted(got)}

    def _pii(self, case: Dict[str, Any]):
        res = get_pii_masker().mask(case["text"], f"eval-{case['case_id']}")
        if case.get("expect_refuse"):
            return (not res.ok), {"ok": res.ok}
        masked = res.masked_text
        leaked = any(v in masked for v in ["张三", "13812345678", "110101201201011234", "S2026001"])
        return res.ok and not leaked, {"hits": len(res.hits)}

    def _sensitive(self, case: Dict[str, Any]):
        hits = self.scanner.scan(case["text"])
        return self.scanner.should_block(hits) == case.get("expect_block", False), \
            {"hits": [h.word for h in hits]}

    # ---------- 汇总 ----------
    def report(self) -> Dict[str, Dict[str, int]]:
        groups: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            g = groups.setdefault(r["type"], {"pass": 0, "fail": 0})
            g["pass" if r["passed"] else "fail"] += 1
        return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="慧学离线评测")
    parser.add_argument("--set", default=str(ROOT / "eval" / "eval_set.jsonl"))
    parser.add_argument("--out", default="", help="逐条结果导出路径（供 badcase 归因）")
    args = parser.parse_args()

    cases = load_cases(Path(args.set))
    ev = Evaluator()
    fails = 0
    for case in cases:
        r = ev.run(case)
        mark = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            fails += 1
        print(f"[{mark}] {r['case_id']:8s} {r['type']:18s} {json.dumps(r['detail'], ensure_ascii=False)[:110]}")

    print("\n=== 分指标汇总 ===")
    for kind, g in sorted(ev.report().items()):
        total = g["pass"] + g["fail"]
        print(f"{kind:20s} {g['pass']}/{total} = {g['pass'] / total:.0%}")
    print(f"总计: {len(cases) - fails}/{len(cases)} 通过")

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ev.results), encoding="utf-8")
        print(f"逐条结果已导出: {args.out}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
