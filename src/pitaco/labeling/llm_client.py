"""Cached, budget-capped client for OpenRouter (OpenAI-compatible API).

Three guarantees:
1. Cache: identical requests are never paid twice. The cache file is also the spend ledger, so
   the budget survives restarts.
2. Hard cap: before every call, worst-case cost (input estimate + `max_tokens` of output) is
   reserved; if spent + reserved would exceed `max_usd`, the call is refused.
3. Reasoning off: models with "thinking" enabled by default would bill hidden reasoning tokens;
   the per-model `reasoning` setting disables it, and leftover reasoning tokens are reported.
"""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pitaco.config import StrictModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHARS_PER_TOKEN = 3.0  # conservative for Portuguese (real tokenizers average ~3.5–4)

Send = Callable[[dict], Awaitable[dict]]


class ModelSpec(StrictModel):
    id: str
    input_per_m: float
    output_per_m: float
    cache_read_per_m: float = 0.0
    reasoning: dict | None = None
    license: str


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def request_key(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class ResponseCache:
    """Append-only JSONL: one line per paid response."""

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, entry: dict) -> None:
        self._entries[entry["key"]] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def spent_usd(self) -> float:
        return sum(e.get("cost_usd") or 0.0 for e in self._entries.values())


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_s: float
    from_cache: bool


def _openrouter_send(api_key: str, base_url: str, timeout_s: float) -> Send:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=3)

    async def send(payload: dict) -> dict:
        body = dict(payload)
        extra = {k: body.pop(k) for k in ("reasoning", "provider") if k in body}
        response = await client.chat.completions.create(**body, extra_body=extra)
        return response.model_dump()

    return send


class LLMClient:
    def __init__(
        self,
        cache: ResponseCache,
        max_usd: float,
        *,
        send: Send | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        concurrency: int = 8,
        timeout_s: float = 60.0,
    ):
        self.cache = cache
        self.max_usd = max_usd
        self._send = send
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._reserved = 0.0
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def spent_usd(self) -> float:
        return self.cache.spent_usd()

    def _sender(self) -> Send:
        if self._send is None:
            key = self._api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._send = _openrouter_send(key, self._base_url, self._timeout_s)
        return self._send

    @staticmethod
    def worst_case_usd(spec: ModelSpec, messages: list[dict], max_tokens: int) -> float:
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        return (prompt_tokens * spec.input_per_m + max_tokens * spec.output_per_m) / 1e6

    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        payload = {"model": spec.id, "messages": messages, "temperature": temperature}
        payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
            payload["provider"] = {"require_parameters": True}
        if spec.reasoning is not None:
            payload["reasoning"] = spec.reasoning
        key = request_key(payload)

        cached = self.cache.get(key)
        if cached is not None:
            return LLMResponse(**cached["response"], from_cache=True)

        worst = self.worst_case_usd(spec, messages, max_tokens)
        # Check and reserve with no `await` in between, so concurrent tasks cannot overshoot.
        if self.spent_usd + self._reserved + worst > self.max_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} + reserved ${self._reserved:.4f} + "
                f"next ${worst:.4f} > cap ${self.max_usd:.2f}"
            )
        self._reserved += worst
        try:
            async with self._semaphore:
                start = time.perf_counter()
                raw = await self._sender()(payload)
                latency = time.perf_counter() - start
        finally:
            self._reserved -= worst

        response = _parse_response(raw, spec, latency)
        self.cache.put(
            {
                "key": key,
                "model": spec.id,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "cost_usd": response.cost_usd,
                "response": {k: v for k, v in response.__dict__.items() if k != "from_cache"},
            }
        )
        return response


def _parse_response(raw: dict, spec: ModelSpec, latency_s: float) -> LLMResponse:
    usage = raw.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or 0
    output_tokens = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    cost = usage.get("cost")
    if cost is None:  # fall back to list prices if the provider omits the cost
        cost = (
            (input_tokens - cached) * spec.input_per_m
            + cached * spec.cache_read_per_m
            + output_tokens * spec.output_per_m
        ) / 1e6
    content = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
        cost_usd=float(cost),
        latency_s=latency_s,
        from_cache=False,
    )
