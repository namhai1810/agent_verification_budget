from __future__ import annotations

import os

from agent_verify.config import BudgetConfig, ModelConfig, ProviderConfig


class BudgetError(RuntimeError):
    pass


class BudgetManager:
    def __init__(
        self,
        config: BudgetConfig,
        provider: ProviderConfig,
        already_spent_usd: float = 0.0,
    ) -> None:
        self.config = config
        self.provider = provider
        self.spent_usd = already_spent_usd
        self.reserved_usd = 0.0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Conservative language-independent approximation for a pre-call check.
        return max(1, (len(text.encode("utf-8")) + 2) // 3)

    def authorize(self, model: ModelConfig, serialized_prompt: str) -> float:
        if model.provider != "openai_compatible":
            return 0.0
        if not self.config.allow_paid_calls or os.getenv("AGENT_VERIFY_ALLOW_SPEND") != "YES":
            raise BudgetError(
                "Paid/model API call blocked. Set budget.allow_paid_calls=true and "
                "AGENT_VERIFY_ALLOW_SPEND=YES only after reviewing MAX_SPEND_USD."
            )
        api_key_env = model.api_key_env or self.provider.api_key_env
        if not os.getenv(api_key_env):
            raise BudgetError(f"Missing credential environment variable: {api_key_env}")
        input_tokens = self.estimate_tokens(serialized_prompt)
        upper_bound = (
            input_tokens * model.input_cost_per_million
            + model.max_output_tokens * model.output_cost_per_million
        ) / 1_000_000
        projected = self.spent_usd + self.reserved_usd + upper_bound
        if projected > self.config.max_spend_usd + 1e-12:
            raise BudgetError(
                f"Pre-call upper bound ${projected:.6f} exceeds MAX_SPEND_USD "
                f"${self.config.max_spend_usd:.6f}."
            )
        self.reserved_usd += upper_bound
        return upper_bound

    def settle(self, reservation: float, actual_cost: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - reservation)
        self.spent_usd += actual_cost
