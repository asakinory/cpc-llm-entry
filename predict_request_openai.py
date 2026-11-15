"""Build one choice-prediction request, send it, and score the A/B answer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NamedTuple

from experiment_config import PRICING, Condition


class ChoiceResult(NamedTuple):
    answer: str  # "A" or "B"
    log_likelihood_a: float | None
    log_likelihood_b: float | None
    cost: float | None
    condition: Condition
    model: str  # the exact model name reported by the API


def request_body(prompt: str, condition: Condition) -> dict[str, Any]:
    """Return the Chat Completions body used for both live and batch requests."""
    return {
        "model": condition.model,
        "service_tier": "default",
        "messages": [
            {"role": "developer", "content": condition.instruction_text},
            {"role": "user", "content": prompt},
        ],
        "reasoning_effort": condition.effort,
        "logprobs": True,
        "top_logprobs": 5,
    }


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _log_one_minus_exp(log_probability: float) -> float:
    """Return log(1 - exp(x)) stably for x <= 0."""
    log_probability = min(log_probability, 0.0)
    if log_probability == 0.0:
        return -math.inf
    if log_probability < math.log(0.5):
        return math.log1p(-math.exp(log_probability))
    return math.log(-math.expm1(log_probability))


def _complete_binary_log_likelihoods(
    log_a: float | None, log_b: float | None
) -> tuple[float | None, float | None]:
    """Normalize returned A/B mass, or infer a missing binary complement."""
    if log_a is not None and log_b is not None:
        normalizer = _logsumexp([log_a, log_b])
        assert normalizer is not None
        return log_a - normalizer, log_b - normalizer
    if log_a is not None:
        log_a = min(log_a, 0.0)
        return log_a, _log_one_minus_exp(log_a)
    if log_b is not None:
        log_b = min(log_b, 0.0)
        return _log_one_minus_exp(log_b), log_b
    return None, None


def _answer_log_likelihoods(sampled: object) -> tuple[float | None, float | None]:
    sampled_token = str(_field(sampled, "token", ""))
    if sampled_token.strip().upper() not in {"A", "B"}:
        return None, None

    candidates = list(_field(sampled, "top_logprobs", []) or [])
    candidates.append(sampled)
    by_answer: dict[str, list[float]] = {"A": [], "B": []}
    seen: set[tuple[str, float]] = set()
    for candidate in candidates:
        token = str(_field(candidate, "token", ""))
        answer = token.strip().upper()
        logprob = _field(candidate, "logprob")
        if answer not in by_answer or not isinstance(logprob, (int, float)):
            continue
        key = (token, float(logprob))
        if key not in seen:
            seen.add(key)
            by_answer[answer].append(float(logprob))

    return _complete_binary_log_likelihoods(
        _logsumexp(by_answer["A"]), _logsumexp(by_answer["B"])
    )


def choice_log_likelihoods(response: object) -> tuple[float | None, float | None]:
    """Return binary-normalized A/B log likelihoods from output-token scores.

    Accepts a Chat Completions response object or its JSON body. Returns
    ``(None, None)`` when the response carries no token log-probabilities.
    """
    choices = _field(response, "choices", []) or []
    if not choices:
        return None, None
    logprobs = _field(_field(choices[0], "logprobs"), "content", []) or []
    for sampled in logprobs:
        result = _answer_log_likelihoods(sampled)
        if result != (None, None):
            return result
    return None, None


def request_cost(
    usage: object, pricing: dict[str, Any] | None, multiplier: float = 1.0
) -> float | None:
    """Return the request cost in USD, or None when usage or prices are unknown."""
    if not usage or not pricing:
        return None

    input_tokens = _field(usage, "prompt_tokens", 0) or 0
    output_tokens = _field(usage, "completion_tokens", 0) or 0
    details = _field(usage, "prompt_tokens_details") or {}
    cached_tokens = _field(details, "cached_tokens", 0) or 0
    cache_write_tokens = _field(details, "cache_write_tokens", 0) or 0
    uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)

    long_context = input_tokens > pricing["long_context_threshold"]
    input_cost = (2.0 if long_context else 1.0) * (
        uncached_tokens * pricing["input"]
        + cached_tokens * pricing["cached_input"]
        + cache_write_tokens * pricing["cache_write"]
    )
    output_cost = (1.5 if long_context else 1.0) * output_tokens * pricing["output"]
    return multiplier * (input_cost + output_cost) / 1_000_000


def choice_request(prompt: str, condition: Condition) -> ChoiceResult:
    """Send one live request and return the A/B answer with its likelihoods."""
    from dotenv import load_dotenv
    from openai import OpenAI

    # The repository's .env is authoritative; otherwise a stale shell-level
    # OPENAI_API_KEY silently takes precedence because dotenv defaults to
    # override=False.
    load_dotenv(Path(__file__).with_name(".env"), override=True)

    response = OpenAI().chat.completions.create(**request_body(prompt, condition))

    output_text = response.choices[0].message.content or ""
    answer = output_text.strip().upper()
    if answer not in {"A", "B"}:
        raise ValueError(f"Expected exactly A or B, got {output_text!r}")
    log_likelihood_a, log_likelihood_b = choice_log_likelihoods(response)
    return ChoiceResult(
        answer=answer,
        log_likelihood_a=log_likelihood_a,
        log_likelihood_b=log_likelihood_b,
        cost=request_cost(response.usage, PRICING.get(condition.model)),
        condition=condition,
        model=response.model,
    )
