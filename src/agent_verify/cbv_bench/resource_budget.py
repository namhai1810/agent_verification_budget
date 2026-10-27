"""Hard experimental Teacher-call/token guard, separate from monetary API budget."""

from __future__ import annotations

from dataclasses import dataclass


class CollectionBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherReservation:
    estimated_upper_tokens: int


class CollectionBudget:
    def __init__(
        self,
        *,
        max_calls: int | None,
        max_tokens: int | None,
        used_calls: int = 0,
        used_tokens: int = 0,
    ) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.used_calls = used_calls
        self.used_tokens = used_tokens
        self.reserved_tokens = 0

    def reserve(self, estimated_upper_tokens: int) -> TeacherReservation:
        if self.max_calls is not None and self.used_calls + 1 > self.max_calls:
            raise CollectionBudgetExhausted("Teacher call budget exhausted")
        if (
            self.max_tokens is not None
            and self.used_tokens + self.reserved_tokens + estimated_upper_tokens > self.max_tokens
        ):
            raise CollectionBudgetExhausted("Teacher token budget cannot safely admit the next call")
        self.reserved_tokens += estimated_upper_tokens
        return TeacherReservation(estimated_upper_tokens=estimated_upper_tokens)

    def cancel(self, reservation: TeacherReservation) -> None:
        self.reserved_tokens -= reservation.estimated_upper_tokens

    def settle(self, reservation: TeacherReservation, actual_tokens: int) -> None:
        self.reserved_tokens -= reservation.estimated_upper_tokens
        self.used_calls += 1
        self.used_tokens += actual_tokens
        if self.max_tokens is not None and self.used_tokens > self.max_tokens:
            raise RuntimeError("Provider usage exceeded the conservative Teacher-token reservation")
