"""Official LiveCodeBench evaluator bridge wrapped by the project OS sandbox."""

from __future__ import annotations

import ast
from functools import partial
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_verify.config import EvaluationConfig
from agent_verify.evaluation.humaneval_runner import SandboxUnavailable
from agent_verify.livecodebench_dataset import OFFICIAL_EVALUATOR_COMMIT
from agent_verify.logging import JsonlStore, content_id
from agent_verify.schemas import EvaluationResult, LiveCodeBenchPrivateCase

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_FORMAT_ERROR = "FORMAT_ERROR"
STATUS_EVALUATOR_ERROR = "EVALUATOR_ERROR"


class OfficialEvaluatorUnavailable(RuntimeError):
    pass


class EvaluatorInfrastructureError(RuntimeError):
    """An evaluator/infrastructure failure that must not become model FAIL."""


class LiveCodeBenchRunner:
    """Evaluate one generation via upstream ``testing_util.run_test`` in isolation."""

    def __init__(
        self,
        config: EvaluationConfig,
        evaluation_log: JsonlStore,
        official_repo: Path,
    ) -> None:
        self.config = config
        self.evaluation_log = evaluation_log
        self.official_repo = official_repo.resolve()
        self.worker_path = Path(__file__).with_name("worker.py").resolve()
        self._sandbox_checked = False
        self._official_checked = False

    def check_official_evaluator(self) -> None:
        if self._official_checked:
            return
        util = self.official_repo / "lcb_runner" / "evaluation" / "testing_util.py"
        if not util.is_file():
            raise OfficialEvaluatorUnavailable(
                f"Official LiveCodeBench checkout missing at {self.official_repo}. "
                "Run `bash scripts/setup_livecodebench.sh`."
            )
        completed = subprocess.run(
            ["git", "-C", str(self.official_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        commit = completed.stdout.strip()
        if completed.returncode != 0 or commit != OFFICIAL_EVALUATOR_COMMIT:
            raise OfficialEvaluatorUnavailable(
                f"Official evaluator revision mismatch: expected {OFFICIAL_EVALUATOR_COMMIT}, "
                f"got {commit or 'unknown'}"
            )
        self._official_checked = True

    def check_sandbox(self) -> None:
        if self._sandbox_checked:
            return
        probe = subprocess.run(
            [sys.executable, "-I", str(self.worker_path), "--probe-isolation"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            preexec_fn=self._set_limits,
            env=self._clean_env(Path("/tmp")),
        )
        if probe.returncode != 0 or '"isolation_ok": true' not in probe.stdout.lower():
            detail = (probe.stderr or probe.stdout).strip()
            raise SandboxUnavailable(f"Landlock/seccomp probe failed: {detail}")
        self._sandbox_checked = True

    def evaluate(
        self,
        *,
        run_id: str,
        case: LiveCodeBenchPrivateCase,
        code: str,
        branch: str,
    ) -> EvaluationResult:
        self.check_official_evaluator()
        self.check_sandbox()
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return self._record(run_id, case.task_id, branch, code, False, STATUS_FORMAT_ERROR, 0.0)

        payload = {
            "mode": "livecodebench",
            "code": code,
            "input_output": case.input_output,
            "per_test_timeout": min(6, self.config.timeout_seconds),
            "official_repo": str(self.official_repo),
        }
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="agent-verify-lcb-", delete=False, encoding="utf-8"
        )
        output_handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="agent-verify-lcb-output-", delete=False, encoding="utf-8"
        )
        payload_path = Path(handle.name)
        output_path = Path(output_handle.name)
        try:
            json.dump(payload, handle)
            handle.close()
            os.chmod(payload_path, 0o600)
            started = time.perf_counter()
            worker_timeout = self.config.timeout_seconds * max(1, case.test_count) + 5
            try:
                process = subprocess.run(
                    [sys.executable, "-I", str(self.worker_path), str(payload_path)],
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=worker_timeout,
                    check=False,
                    # The official evaluator enforces the per-test timeout. Its
                    # wrapper must have enough CPU budget to iterate over all
                    # tests and serialize the model outcome.
                    preexec_fn=partial(self._set_limits, cpu_seconds=worker_timeout + 2),
                    env=self._clean_env(payload_path.parent),
                )
                output_handle.flush()
                latency = time.perf_counter() - started
                if process.returncode != 0:
                    status, passed = self._worker_exit_status(process.returncode), False
                else:
                    lines = [
                        line
                        for line in output_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip()
                    ]
                    worker_result = json.loads(lines[-1]) if lines else {}
                    status = str(worker_result.get("status", STATUS_EVALUATOR_ERROR))
                    passed = bool(worker_result.get("passed", False))
            except subprocess.TimeoutExpired:
                latency = time.perf_counter() - started
                status, passed = STATUS_TIMEOUT, False
            result = self._record(run_id, case.task_id, branch, code, passed, status, latency)
            return self._require_model_outcome(result)
        except EvaluatorInfrastructureError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            latency = time.perf_counter() - started if "started" in locals() else 0.0
            result = self._record(
                run_id,
                case.task_id,
                branch,
                code,
                False,
                f"{STATUS_EVALUATOR_ERROR}:{type(exc).__name__}",
                latency,
            )
            return self._require_model_outcome(result)
        finally:
            output_handle.close()
            payload_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    @staticmethod
    def _require_model_outcome(result: EvaluationResult) -> EvaluationResult:
        if result.status.startswith(STATUS_EVALUATOR_ERROR) or result.status.startswith(
            "sandbox_setup_failed"
        ):
            raise EvaluatorInfrastructureError(
                f"Evaluator infrastructure failed for {result.task_id}: {result.status}"
            )
        return result

    
    def _worker_exit_status(returncode: int) -> str:
        """Map resource-limit exits without hiding genuine evaluator failures."""

        if returncode in {-signal.SIGXCPU, -signal.SIGKILL}:
            return STATUS_TIMEOUT
        return f"{STATUS_EVALUATOR_ERROR}:worker_exit_{returncode}"

    def _record(
        self,
        run_id: str,
        task_id: str,
        branch: str,
        code: str,
        passed: bool,
        status: str,
        latency: float,
    ) -> EvaluationResult:
        identity = {"run_id": run_id, "task_id": task_id, "branch": branch, "code": code}
        result = EvaluationResult(
            evaluation_id=content_id("eval", identity),
            run_id=run_id,
            task_id=task_id,
            branch=branch,
            passed=passed,
            status=status,
            latency_seconds=latency,
            sandbox="official-lcb-run_test+landlock+seccomp+rlimits",
        )
        self.evaluation_log.append(result)
        return result

    @staticmethod
    def _clean_env(home: Path) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
        }

    def _set_limits(self, cpu_seconds: int | None = None) -> None:
        memory_bytes = self.config.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_limit = max(1, cpu_seconds or self.config.timeout_seconds)
        # A distinct hard limit lets SIGXCPU surface before SIGKILL. The
        # parent maps either resource-limit signal to a model TIMEOUT.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        resource.setrlimit(resource.RLIMIT_NPROC, (self.config.max_processes, self.config.max_processes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
