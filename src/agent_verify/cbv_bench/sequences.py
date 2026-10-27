"""Action-sequence helpers for the fixed three-checkpoint workflow."""

from __future__ import annotations

from typing import Literal

Action = Literal["S", "V"]
ActionSequence = Literal["SSS", "VSS", "SVS", "SSV", "VVS", "VSV", "SVV", "VVV"]

LEVEL1_SEQUENCES: tuple[ActionSequence, ...] = ("SSS", "VSS", "SVS", "SSV")
LEVEL2_SEQUENCES: tuple[ActionSequence, ...] = (
    "SSS",
    "VSS",
    "SVS",
    "SSV",
    "VVS",
    "VSV",
    "SVV",
    "VVV",
)


def required_sequences(level: int) -> tuple[ActionSequence, ...]:
    if level == 1:
        return LEVEL1_SEQUENCES
    if level == 2:
        return LEVEL2_SEQUENCES
    raise ValueError("CBV-Bench level must be 1 or 2")


def required_prefixes(level: int) -> tuple[str, ...]:
    """Return unique non-empty prefixes needed to construct a level."""
    prefixes = {sequence[:depth] for sequence in required_sequences(level) for depth in (1, 2, 3)}
    return tuple(sorted(prefixes, key=lambda value: (len(value), value)))


def teacher_call_count(level: int) -> int:
    """Number of unique V edges in the shared prefix tree."""
    prefixes = required_prefixes(level)
    return sum(prefix.endswith("V") for prefix in prefixes)
