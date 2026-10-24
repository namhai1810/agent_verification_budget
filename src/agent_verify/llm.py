from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agent_verify.budget import BudgetManager
from agent_verify.config import ModelConfig, ProviderConfig
from agent_verify.logging import JsonlStore, canonical_json
from agent_verify.schemas import InvocationContext, InvocationRecord, Message

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMResult:
    value: BaseModel
    record: InvocationRecord


class LLMClient:
    def __init__(
        self,
        provider_config: ProviderConfig,
        budget: BudgetManager,
        invocation_log: JsonlStore,
    ) -> None:
        self.provider_config = provider_config
        self.budget = budget
        self.invocation_log = invocation_log

    def invoke(
        self,
        *,
        model: ModelConfig,
        messages: list[Message],
        output_schema: type[T],
        context: InvocationContext,
        mock_factory: Callable[[], T],
    ) -> LLMResult:
        serialized = canonical_json([message.model_dump() for message in messages])
        prompt_hash = hashlib.sha256(serialized.encode()).hexdigest()
        invocation_id = f"inv_{uuid.uuid4().hex}"
        started = time.perf_counter()
        raw_output: str | None = None
        attempt_count = 0
        reservation = 0.0
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        attempts: list[dict] = []
        try:
            if model.provider == "mock":
                attempt_count = 1
                parsed = mock_factory()
                raw_output = parsed.model_dump_json()
                input_tokens = BudgetManager.estimate_tokens(serialized)
                output_tokens = BudgetManager.estimate_tokens(raw_output)
                attempts.append(
                    {
                        "attempt": 1,
                        "status": "success",
                        "raw_output": raw_output,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": 0.0,
                    }
                )
            elif model.provider == "openai_compatible":
                parsed = None
                last_error: Exception | None = None
                for attempt_count in range(1, self.provider_config.max_retries + 2):
                    attempt_started = time.perf_counter()
                    attempt_raw: str | None = None
                    attempt_input_tokens = 0
                    attempt_output_tokens = 0
                    attempt_cost = 0.0
                    reservation = self.budget.authorize(model, serialized)
                    try:
                        attempt_raw, attempt_input_tokens, attempt_output_tokens = self._call_api(
                            model, messages, output_schema, context.seed
                        )
                        raw_output = attempt_raw
                        attempt_cost = (
                            attempt_input_tokens * model.input_cost_per_million
                            + attempt_output_tokens * model.output_cost_per_million
                        ) / 1_000_000
                        self.budget.settle(reservation, attempt_cost)
                        reservation = 0.0
                        input_tokens += attempt_input_tokens
                        output_tokens += attempt_output_tokens
                        cost += attempt_cost
                        parsed = output_schema.model_validate(self._parse_json(attempt_raw))
                        attempts.append(
                            {
                                "attempt": attempt_count,
                                "status": "success",
                                "raw_output": attempt_raw,
                                "input_tokens": attempt_input_tokens,
                                "output_tokens": attempt_output_tokens,
                                "cost_usd": attempt_cost,
                                "latency_seconds": time.perf_counter() - attempt_started,
                            }
                        )
                        break
                    except (
                        httpx.HTTPError,
                        KeyError,
                        IndexError,
                        ValueError,
                        ValidationError,
                    ) as exc:
                        if reservation:
                            self.budget.settle(reservation, 0.0)
                            reservation = 0.0
                        attempts.append(
                            {
                                "attempt": attempt_count,
                                "status": "failed",
                                "raw_output": attempt_raw,
                                "input_tokens": attempt_input_tokens,
                                "output_tokens": attempt_output_tokens,
                                "cost_usd": attempt_cost,
                                "latency_seconds": time.perf_counter() - attempt_started,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        last_error = exc
                        if attempt_count > self.provider_config.max_retries:
                            raise
                        time.sleep(min(2 ** (attempt_count - 1), 4))
                if parsed is None:
                    raise RuntimeError("LLM invocation ended without a parsed response") from last_error
            else:
                raise ValueError(f"Provider {model.provider!r} is not valid for an LLM role")

            record = InvocationRecord(
                invocation_id=invocation_id,
                context=context,
                llm_config=model,
                prompt_hash=prompt_hash,
                messages=messages,
                raw_output=raw_output,
                parsed_output=parsed.model_dump(mode="json"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_seconds=time.perf_counter() - started,
                attempt_count=attempt_count,
                attempts=attempts,
            )
            self.invocation_log.append(record)
            return LLMResult(parsed, record)
        except Exception as exc:
            if reservation:
                self.budget.settle(reservation, 0.0)
            record = InvocationRecord(
                invocation_id=invocation_id,
                context=context,
                llm_config=model,
                prompt_hash=prompt_hash,
                messages=messages,
                raw_output=raw_output,
                parsed_output=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_seconds=time.perf_counter() - started,
                attempt_count=attempt_count,
                attempts=attempts,
                failure=f"{type(exc).__name__}: {exc}",
            )
            self.invocation_log.append(record)
            raise

    def _call_api(
        self,
        model: ModelConfig,
        messages: list[Message],
        output_schema: type[T],
        seed: int,
    ) -> tuple[str, int, int]:
        schema = output_schema.model_json_schema()
        if self.provider_config.response_format == "json_schema":
            response_format: dict = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        payload = {
            "model": model.model_id,
            "messages": [message.model_dump() for message in messages],
            "temperature": model.temperature,
            "top_p": model.top_p,
            "max_tokens": model.max_output_tokens,
            "seed": seed,
            "response_format": response_format,
        }
        import os

        api_key_env = model.api_key_env or self.provider_config.api_key_env
        headers = {
            "Authorization": f"Bearer {os.environ[api_key_env]}",
            "Content-Type": "application/json",
        }
        base_url = model.base_url or self.provider_config.base_url
        url = base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.provider_config.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        raw = body["choices"][0]["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("Expected string content from chat completion")
        usage = body.get("usage") or {}
        input_tokens = int(
            usage.get("prompt_tokens") or BudgetManager.estimate_tokens(canonical_json(payload["messages"]))
        )
        output_tokens = int(usage.get("completion_tokens") or BudgetManager.estimate_tokens(raw))
        return raw, input_tokens, output_tokens

    @staticmethod
    def _parse_json(raw: str) -> dict:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1]
            if candidate.endswith("```"):
                candidate = candidate[:-3]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(candidate[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Structured response must be a JSON object")
        return parsed
