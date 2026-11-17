"""PII 脱敏器测试：占位符替换、映射还原、不可靠即拒答。"""
from __future__ import annotations

import pytest

from backend.app.safety.pii_masker import PiiMasker


@pytest.fixture()
def masker() -> PiiMasker:
    return PiiMasker()


def test_mask_phone_and_id(masker: PiiMasker):
    res = masker.mask("我的手机是13812345678，身份证110101201201011234", "s1")
    assert res.ok
    assert "13812345678" not in res.masked_text
    assert "110101201201011234" not in res.masked_text
    assert "[PII:手机号:1]" in res.masked_text


def test_mask_student_id_and_name(masker: PiiMasker):
    res = masker.mask("学生李小明，学号 S2026001，问一道题", "s2")
    assert res.ok
    assert "李小明" not in res.masked_text
    assert "S2026001" not in res.masked_text
    assert "PII:姓名" in res.masked_text and "PII:学号" in res.masked_text


def test_restore_roundtrip(masker: PiiMasker):
    original = "学生王小红学号 S2026002 的手机13800001111需要验证"
    res = masker.mask(original, "s3")
    assert res.ok
    restored = masker.restore(res.masked_text, "s3")
    assert restored == original


def test_unreliable_long_digits_refuse(masker: PiiMasker):
    """超长数字串无法可靠定位边界 -> ok=False，调用方必须拒答。"""
    res = masker.mask("1234567890123456789012 是什么", "s4")
    assert not res.ok
    assert res.reason


def test_contains_pii(masker: PiiMasker):
    assert masker.contains_pii("联系方式13812345678")
    assert not masker.contains_pii("3千米等于多少米")


def test_same_value_reuses_placeholder(masker: PiiMasker):
    res = masker.mask("13812345678和13812345678", "s5")
    assert res.masked_text.count("[PII:手机号:1]") == 2


def test_drop_session_clears_mapping(masker: PiiMasker):
    masker.mask("手机13812345678", "s6")
    masker.drop_session("s6")
    assert masker.restore("[PII:手机号:1]", "s6") == "[PII:手机号:1]"
