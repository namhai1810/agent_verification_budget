#!/usr/bin/env python3
"""Fail-fast live probe proving that vLLM applies thinking_token_budget."""

from __future__ import annotations

import argparse
import json
import os

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--probe-budget", type=int, default=64)
    args = parser.parse_args()
    if args.probe_budget < 1:
        raise SystemExit("--probe-budget must be positive")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Think carefully about whether 123456789 is divisible by 9. "
                    "After reasoning, return only the word YES or NO."
                ),
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 256,
        "thinking_token_budget": args.probe_budget,
        "include_reasoning": True,
        "seed": 20261118,
    }
    key = os.getenv("VLLM_API_KEY", "local-vllm")
    response = httpx.post(
        args.base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=300.0,
    )
    if response.status_code >= 400:
        raise SystemExit(
            "thinking_token_budget probe was rejected; refusing to continue:\n"
            + response.text
        )
    body = response.json()
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    choice = (body.get("choices") or [{}])[0]
    final_content = (choice.get("message") or {}).get("content")
    if not isinstance(reasoning_tokens, int):
        raise SystemExit(
            "Server did not report reasoning_tokens; budget enforcement cannot be verified."
        )
    if reasoning_tokens > args.probe_budget:
        raise SystemExit(
            f"Budget NOT enforced: reasoning_tokens={reasoning_tokens} > {args.probe_budget}"
        )
    if choice.get("finish_reason") == "length" or not str(final_content or "").strip():
        raise SystemExit("Probe did not preserve a non-truncated final answer.")
    print(
        json.dumps(
            {
                "supported": True,
                "enforced": True,
                "probe_budget": args.probe_budget,
                "reasoning_tokens": reasoning_tokens,
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": choice.get("finish_reason"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
