"""模型网关：OpenAI 兼容协议封装，不绑定任何厂商。

与常见模板项目的差异点：
1. 场景化调用 —— 每次请求必须声明 scene（router/lesson/qa/grading/profile），
   网关按场景统计调用量并独立限流，便于成本核算与灰度对比；
2. 双模降级 —— 未配置密钥时自动切换 ``MockBackend``，返回确定性结果，
   调用方可通过 ``mock_example`` 注入本场景的确定性算法输出，
   保证无密钥环境下备课/答疑/批改/学情全链路与单测可离线跑通；
3. 语义缓存 —— 以 (模型, 场景, 规范化消息) 的 SHA256 为键做进程内 LRU 缓存，
   与 db.cache 的"镜像双缓存"分层解耦：这里只缓存原始补全，不做业务写回；
4. 令牌桶限流 + 指数退避重试，仅对 429/5xx/网络错误重试，业务错误快速失败。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from backend.app.config import Settings, get_settings

ALLOWED_SCENES = ("router", "lesson", "qa", "grading", "profile")


class GatewayError(RuntimeError):
    """网关不可恢复错误（业务错误 / 重试耗尽）。"""


@dataclass
class GatewayResult:
    """一次补全调用的结构化结果。"""

    text: str
    scene: str
    model: str
    cached: bool = False
    latency_ms: int = 0
    usage: Dict[str, int] = field(default_factory=dict)


class TokenBucket:
    """令牌桶限流器：按场景共享一个桶，QPS 由配置决定。"""

    def __init__(self, qps: float, capacity: int = 20) -> None:
        self.qps = max(qps, 0.1)
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def acquire(self) -> None:
        """阻塞式取令牌；mock 模式下几乎瞬时，不会拖慢测试。"""
        while True:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.qps)
            self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return
            time.sleep(0.01)


class LRUSemanticCache:
    """进程内 LRU 语义缓存，键为规范化消息的哈希。"""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._store: "OrderedDict[str, str]" = OrderedDict()

    @staticmethod
    def make_key(model: str, scene: str, messages: List[Dict[str, str]]) -> str:
        def _norm(m: Dict[str, str]) -> str:
            content = re.sub(r"\s+", " ", m.get("content", "")).strip()
            return f"{m.get('role', 'user')}:{content}"

        normalized = "|".join(_norm(m) for m in messages)
        digest = hashlib.sha256(f"{model}::{scene}::{normalized}".encode("utf-8")).hexdigest()
        return digest

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)


class MockBackend:
    """确定性 mock 后端。

    优先返回调用方注入的 ``mock_example``（各子图在 mock 模式下提供本场景
    的确定性算法结果，例如基于关键词重合度的主观题评分）；未注入时
    返回带引用格式的通用话术，保证下游引用校验节点仍可走通。
    """

    _GENERIC = {
        "qa": "根据《慧学示例教材》三年级上册第 12 页的说明：先统一单位再比较大小。【来源: demo-math-3a | 第12页】",
        "lesson": "已完成备课草稿：围绕核心概念设计 3 个教学环节与 2 道随堂练习，待教师复核。",
        "grading": "本题解题步骤完整，结论正确，按评分要点给分。",
        "profile": "该生在「单位换算」分支掌握度偏低，建议安排针对性练习。",
        "router": "ok",
    }

    def complete(self, scene: str, messages: List[Dict[str, str]],
                 mock_example: Optional[Any], json_mode: bool) -> str:
        if mock_example is not None:
            if isinstance(mock_example, str):
                return mock_example
            return json.dumps(mock_example, ensure_ascii=False)
        if json_mode:
            # 未注入示例时返回结构合法的最小 JSON，避免解析崩溃
            return json.dumps({"result": self._GENERIC.get(scene, "ok"), "confidence": 0.5},
                              ensure_ascii=False)
        return self._GENERIC.get(scene, "ok")


class ModelGateway:
    """厂商无关的聊天补全网关。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.cache = LRUSemanticCache()
        self.bucket = TokenBucket(self.settings.rate_limit_qps)
        self.mock = MockBackend()
        self._stats: Dict[str, int] = {s: 0 for s in ALLOWED_SCENES}

    # ---------- 对外主入口 ----------
    def chat(self, messages: List[Dict[str, str]], *, scene: str,
             temperature: Optional[float] = None, max_tokens: int = 1024,
             json_mode: bool = False, mock_example: Optional[Any] = None) -> GatewayResult:
        """同步补全调用。scene 必须在白名单内，便于分场景统计。"""
        if scene not in ALLOWED_SCENES:
            raise GatewayError(f"未知场景: {scene}")
        self._stats[scene] += 1
        key = LRUSemanticCache.make_key(self.settings.chat_model, scene, messages)
        hit = self.cache.get(key)
        if hit is not None:
            return GatewayResult(text=hit, scene=scene, model=self.settings.chat_model, cached=True)

        if self.settings.mock_mode:
            text = self.mock.complete(scene, messages, mock_example, json_mode)
        else:
            text = self._call_remote(messages, temperature, max_tokens, json_mode)

        self.cache.put(key, text)
        return GatewayResult(text=text, scene=scene, model=self.settings.chat_model)

    def chat_json(self, messages: List[Dict[str, str]], *, scene: str,
                  mock_example: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        """要求 JSON 输出的便捷方法：解析失败自动纠错重试一次。"""
        result = self.chat(messages, scene=scene, json_mode=True,
                           mock_example=mock_example, **kwargs)
        try:
            return json.loads(_strip_code_fence(result.text))
        except json.JSONDecodeError:
            repair = list(messages) + [{
                "role": "user",
                "content": "上一次输出不是合法 JSON。请仅输出一个 JSON 对象，不要任何解释文字。",
            }]
            fixed = self.chat(repair, scene=scene, json_mode=True, **kwargs)
            try:
                return json.loads(_strip_code_fence(fixed.text))
            except json.JSONDecodeError as exc:  # 快速失败，交由上层降级
                raise GatewayError(f"模型输出无法解析为 JSON: {fixed.text[:200]}") from exc

    async def stream_chat(self, messages: List[Dict[str, str]], *, scene: str,
                          mock_example: Optional[Any] = None) -> AsyncIterator[str]:
        """流式补全（供 SSE 端点使用）。mock 模式按标点切块模拟吐字。"""
        if self.settings.mock_mode:
            text = self.mock.complete(scene, messages, mock_example, False)
            for piece in re.split(r"(?<=[，。！？；])", text):
                if piece:
                    await asyncio.sleep(0.01)
                    yield piece
            return
        result = await asyncio.to_thread(
            self.chat, messages, scene=scene)
        for piece in re.split(r"(?<=[，。！？；])", result.text):
            if piece:
                yield piece

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # ---------- 真实远端调用 ----------
    def _call_remote(self, messages: List[Dict[str, str]], temperature: Optional[float],
                     max_tokens: int, json_mode: bool) -> str:
        import httpx  # 延迟导入：mock 环境无需该依赖

        payload: Dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}

        self.bucket.acquire()
        last_err: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                with httpx.Client(timeout=60) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise _RetryableHTTP(resp.status_code)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (_RetryableHTTP, httpx.TransportError) as exc:
                last_err = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.2)
        raise GatewayError(f"网关重试耗尽: {last_err}")


class _RetryableHTTP(Exception):
    """标记可重试的 HTTP 状态。"""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"retryable http status {status_code}")
        self.status_code = status_code


def _strip_code_fence(text: str) -> str:
    """剥掉 ```json 围栏，兼容部分网关不遵守 response_format 的情况。"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    return match.group(1) if match else text.strip()


_gateway_singleton: Optional[ModelGateway] = None


def get_gateway() -> ModelGateway:
    """进程级网关单例。"""
    global _gateway_singleton
    if _gateway_singleton is None:
        _gateway_singleton = ModelGateway()
    return _gateway_singleton
