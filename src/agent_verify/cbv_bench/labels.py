"""Counterfactual labels computed only after evaluation, never used during generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from agent_verify.cbv_bench.schemas import EffectRecord
from agent_verify.cbv_bench.sequences import LEVEL1_SEQUENCES, LEVEL2_SEQUENCES


def _label(delta: int, interaction: bool = False) -> str:
    if interaction:
        return "Positive" if delta > 0 else "Negative" if delta < 0 else "Zero"
    return "Helpful" if delta > 0 else "Harmful" if delta < 0 else "NoChange"


def _effect(dataset: str, task_id: str, rollout: int, level: int, effect_type: str,
            skip: str, verify: str, outcomes: dict[str, dict], checkpoint: int | None = None,
            preceding: str = "", fixed: str = "") -> EffectRecord:
    if outcomes[skip].get("model_pair_fingerprint") != outcomes[verify].get(
        "model_pair_fingerprint"
    ):
        raise ValueError(f"Refusing cross-model-pair contrast: {skip} vs {verify}")
    y_skip, y_verify = int(outcomes[skip]["passed"]), int(outcomes[verify]["passed"])
    gain = y_verify - y_skip
    token_delta = int(outcomes[verify].get("teacher_total_tokens", 0)) - int(
        outcomes[skip].get("teacher_total_tokens", 0)
    )
    return EffectRecord(
        model_pair_id=str(outcomes[skip]["model_pair_id"]),
        model_pair_fingerprint=str(outcomes[skip]["model_pair_fingerprint"]),
        dataset=dataset, task_id=task_id, rollout_index=rollout, level=level,
        effect_type=effect_type, checkpoint=checkpoint, preceding_history=preceding,
        fixed_other_actions=fixed, skip_sequence=skip, verify_sequence=verify,
        y_skip=y_skip, y_verify=y_verify, gain=gain, label=_label(gain),
        teacher_token_delta=token_delta,
    )


def build_effect_records(results_dir: Path, selector: str, level: int) -> Path:
    """Build Level-1 and true history-conditioned Level-2 effects.

    ``selector`` may be a run ID or compatibility fingerprint. Comparing siblings uses the same
    preceding history and identical other actions; Level-1 gains are never arithmetically summed.
    """
    path = results_dir / "trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("status") == "completed" and (
        row.get("run_id") == selector or row.get("compatibility_fingerprint") == selector
    )]
    grouped: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["task_id"], int(row["rollout_index"]))][
            row["action_sequence"]
        ] = row
    effects: list[EffectRecord] = []
    for (dataset, task_id, rollout), outcomes in sorted(grouped.items()):
        if set(LEVEL1_SEQUENCES).issubset(outcomes):
            for checkpoint, verify in ((1, "VSS"), (2, "SVS"), (3, "SSV")):
                effects.append(_effect(dataset, task_id, rollout, 1, "level1", "SSS", verify,
                                       outcomes, checkpoint, "S" * (checkpoint - 1), "S"))
        if level < 2 or not set(LEVEL2_SEQUENCES).issubset(outcomes):
            continue
        pair_fingerprints = {
            str(outcomes[sequence].get("model_pair_fingerprint"))
            for sequence in LEVEL2_SEQUENCES
        }
        if len(pair_fingerprints) != 1:
            raise ValueError(f"Mixed model pairs in {dataset}/{task_id}/rollout-{rollout}")
        # Twelve hypercube edges: action differs only at the named checkpoint.
        for checkpoint in (1, 2, 3):
            position = checkpoint - 1
            for skip in LEVEL2_SEQUENCES:
                if skip[position] != "S":
                    continue
                verify = skip[:position] + "V" + skip[position + 1:]
                effects.append(_effect(
                    dataset, task_id, rollout, 2, "conditional", skip, verify, outcomes,
                    checkpoint, skip[:position], skip[position + 1:],
                ))
        y = {sequence: int(outcomes[sequence]["passed"]) for sequence in LEVEL2_SEQUENCES}
        pairwise = [
            ("cp1_cp2|cp3=S", y["VVS"] - y["VSS"] - y["SVS"] + y["SSS"]),
            ("cp1_cp2|cp3=V", y["VVV"] - y["VSV"] - y["SVV"] + y["SSV"]),
            ("cp1_cp3|cp2=S", y["VSV"] - y["VSS"] - y["SSV"] + y["SSS"]),
            ("cp1_cp3|cp2=V", y["VVV"] - y["VVS"] - y["SVV"] + y["SVS"]),
            ("cp2_cp3|cp1=S", y["SVV"] - y["SVS"] - y["SSV"] + y["SSS"]),
            ("cp2_cp3|cp1=V", y["VVV"] - y["VVS"] - y["VSV"] + y["VSS"]),
        ]
        for name, value in pairwise:
            effects.append(EffectRecord(
                model_pair_id=str(outcomes["SSS"]["model_pair_id"]),
                model_pair_fingerprint=str(outcomes["SSS"]["model_pair_fingerprint"]),
                dataset=dataset, task_id=task_id, rollout_index=rollout, level=2,
                effect_type="pairwise_interaction", gain=value, label=_label(value, True),
                details={"contrast": name},
            ))
        three_way = (
            y["VVV"] - y["VVS"] - y["VSV"] - y["SVV"]
            + y["VSS"] + y["SVS"] + y["SSV"] - y["SSS"]
        )
        effects.append(EffectRecord(
            model_pair_id=str(outcomes["SSS"]["model_pair_id"]),
            model_pair_fingerprint=str(outcomes["SSS"]["model_pair_fingerprint"]),
            dataset=dataset, task_id=task_id, rollout_index=rollout, level=2,
            effect_type="three_way_interaction", gain=three_way,
            label=_label(three_way, True), details={"contrast": "cp1_x_cp2_x_cp3"},
        ))
    output = results_dir / "derived" / selector / "effects.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(record.model_dump_json() + "\n" for record in effects), encoding="utf-8")
    return output
