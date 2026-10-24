from pathlib import Path

from agent_verify.cli import _overridden_config, _parser


def test_run_cli_overrides_task_window() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    args = _parser().parse_args(
        [
            "run",
            "--config",
            str(config_path),
            "--task-offset",
            "24",
            "--task-count",
            "140",
            "--repeated-pairs",
            "1",
        ]
    )

    config = _overridden_config(args)

    assert config.experiment.task_offset == 24
    assert config.experiment.task_count == 140
    assert config.experiment.repeated_pairs == 1
