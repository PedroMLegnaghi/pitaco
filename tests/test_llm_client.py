"""The client is tested with a fake transport: no network, no money."""

import asyncio

import pytest

from pitaco.labeling.llm_client import BudgetExceeded, LLMClient, ModelSpec, ResponseCache

SPEC = ModelSpec(id="fake/model", input_per_m=1.0, output_per_m=2.0, license="MIT")
MESSAGES = [{"role": "user", "content": "Avaliação: chegou rápido"}]


def fake_send(calls: list, cost: float | None = 0.001, reasoning_tokens: int = 0):
    async def send(payload: dict) -> dict:
        calls.append(payload)
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
        if cost is not None:
            usage["cost"] = cost
        return {
            "choices": [{"message": {"content": '{"aspectos": [], "resumo": "ok"}'}}],
            "usage": usage,
        }

    return send


def _complete(client, messages=MESSAGES, max_tokens=100):
    return asyncio.run(
        client.complete(SPEC, messages, temperature=0, max_tokens=max_tokens, response_format=None)
    )


def test_second_identical_request_hits_cache(tmp_path):
    calls = []
    client = LLMClient(ResponseCache(tmp_path / "c.jsonl"), 1.0, send=fake_send(calls))
    first, second = _complete(client), _complete(client)
    assert len(calls) == 1
    assert not first.from_cache and second.from_cache
    assert second.content == first.content


def test_cache_and_spend_survive_restart(tmp_path):
    path = tmp_path / "c.jsonl"
    _complete(LLMClient(ResponseCache(path), 1.0, send=fake_send([])))
    calls = []
    restarted = LLMClient(ResponseCache(path), 1.0, send=fake_send(calls))
    assert restarted.spent_usd == pytest.approx(0.001)
    _complete(restarted)
    assert calls == []


def test_budget_cap_refuses_before_calling(tmp_path):
    calls = []
    client = LLMClient(ResponseCache(tmp_path / "c.jsonl"), 0.0001, send=fake_send(calls))
    with pytest.raises(BudgetExceeded):
        _complete(client, max_tokens=1000)  # worst case 1000 output tokens × $2/M = $0.002
    assert calls == []


def test_cost_falls_back_to_list_prices(tmp_path):
    client = LLMClient(ResponseCache(tmp_path / "c.jsonl"), 1.0, send=fake_send([], cost=None))
    resp = _complete(client)
    assert resp.cost_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1e6)


def test_reasoning_setting_is_sent_and_tokens_reported(tmp_path):
    calls = []
    spec = SPEC.model_copy(update={"reasoning": {"effort": "none"}})
    client = LLMClient(
        ResponseCache(tmp_path / "c.jsonl"), 1.0, send=fake_send(calls, reasoning_tokens=7)
    )
    resp = asyncio.run(
        client.complete(spec, MESSAGES, temperature=0, max_tokens=50, response_format=None)
    )
    assert calls[0]["reasoning"] == {"effort": "none"}
    assert resp.reasoning_tokens == 7


def test_structured_output_requires_supporting_providers(tmp_path):
    calls = []
    client = LLMClient(ResponseCache(tmp_path / "c.jsonl"), 1.0, send=fake_send(calls))
    fmt = {"type": "json_object"}
    asyncio.run(client.complete(SPEC, MESSAGES, temperature=0, max_tokens=50, response_format=fmt))
    assert calls[0]["response_format"] == fmt
    assert calls[0]["provider"] == {"require_parameters": True}


def test_missing_api_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = LLMClient(ResponseCache(tmp_path / "c.jsonl"), 1.0)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _complete(client)
