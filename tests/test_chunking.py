"""切片器测试：父子块、表格结构化、对话轮次。"""
from __future__ import annotations

from backend.app.retrieval.chunking import (
    DialogueTurnChunker, ParentChildChunker, TableStructuralChunker,
)

_MARKDOWN = """# 长度单位

我们学过的长度单位有千米、米、分米、厘米、毫米。1 千米等于 1000 米。

# 换算方法

把高级单位改成低级单位要乘进率。例如 3 千米等于 3000 米。
"""

_TABLE = """| 知识点 | 常见错因 |
| --- | --- |
| 单位换算 | 进率记错 |
| 周长 | 公式混淆 |
"""

_DIALOGUE = """学生：3千米是多少米？
教师：3000米，因为进率是1000。
家长：那54分米等于几米？
老师：5.4米。
"""


def test_parent_child_structure():
    chunks = ParentChildChunker(child_max_chars=40).chunk("doc1", _MARKDOWN)
    parents = [c for c in chunks if c.meta["kind"] == "parent"]
    children = [c for c in chunks if c.meta["kind"] == "child"]
    assert len(parents) == 2
    assert children, "长段落应继续切出子块"
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_child_within_length_limit():
    chunks = ParentChildChunker(child_max_chars=40).chunk("doc1", _MARKDOWN)
    children = [c for c in chunks if c.meta["kind"] == "child"]
    # 允许单句超限（不可再分），但多句窗口不应明显超限
    multi = [c for c in children if c.text.count("。") > 1]
    assert all(len(c.text) <= 80 for c in multi)


def test_table_rows_bind_header():
    chunks = TableStructuralChunker().chunk("doc2", _TABLE)
    assert len(chunks) == 2
    row = chunks[0]
    assert "知识点:单位换算" in row.text and "常见错因:进率记错" in row.text
    assert row.meta["kind"] == "table_row"


def test_dialogue_pairs_question_answer():
    chunks = DialogueTurnChunker().chunk("doc3", _DIALOGUE)
    assert len(chunks) == 2
    first = chunks[0]
    assert "问：3千米是多少米？" in first.text
    assert "答：3000米" in first.text
    # 家长问 + 老师答 也应配对（"老师"与"教师"同义）
    second = chunks[1]
    assert "54分米" in second.text and "答：" in second.text


def test_chunk_ids_unique():
    chunks = ParentChildChunker().chunk("doc1", _MARKDOWN)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
