"""Isolated BigCodeBench evaluator; existing evaluators remain unchanged."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_verify.config import EvaluationConfig
from agent_verify.evaluation.humaneval_runner import SandboxUnavailable
from agent_verify.logging import JsonlStore, content_id
from agent_verify.schemas import BigCodeBenchPrivateCase, EvaluationResult


class BigCodeBenchEvaluatorError(RuntimeError):
    """Infrastructure/schema failure that must not be counted as model incorrect."""


class BigCodeBenchRunner:
    """Run official task tests in the project sandbox without exposing them to models.

    This bridge is suitable for CBV data collection, but it is deliberately reported as
    ``bigcodebench-isolated-tests`` rather than claiming leaderboard equivalence. The official
    package/environment remains responsible for installing task library dependencies.
    """

    def __init__(self, config: EvaluationConfig, evaluation_log: JsonlStore) -> None:
        self.config = config
        self.evaluation_log = evaluation_log
        self.worker_path = Path(__file__).with_name("worker.py").resolve()
        self._sandbox_checked = False

    def check_sandbox(self) -> None:
        if self._sandbox_checked:
            return
        process = subprocess.run(
            [sys.executable, "-I", str(self.worker_path), "--probe-isolation"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            preexec_fn=self._set_limits,
            env=self._clean_env(Path("/tmp")),
        )
        if process.returncode != 0 or '"isolation_ok": true' not in process.stdout.lower():
            raise SandboxUnavailable((process.stderr or process.stdout).strip())
        self._sandbox_checked = True

    @staticmethod
    def check_dependencies(public_tasks: list) -> None:
        libraries = {
            library
            for task in public_tasks
            for library in getattr(task, "libraries", [])
        }
        missing = sorted(
            library
            for library in libraries
            if importlib.util.find_spec(library.split(".")[0]) is None
        )
        if missing:
            preview = ", ".join(missing[:20])
            raise BigCodeBenchEvaluatorError(
                f"Selected BigCodeBench tasks require unavailable libraries: {preview}"
            )

    def evaluate(self, *, run_id: str, case: BigCodeBenchPrivateCase,
                 code: str, branch: str) -> EvaluationResult:
        self.check_sandbox()
        try:
            ast.parse(code)
        except (SyntaxError, TypeError, ValueError):
            return self._record(run_id, case.task_id, branch, code, False, "FORMAT_ERROR", 0.0)
        payload = {
            "mode": "bigcodebench",
            "code": code,
            "code_prompt": case.code_prompt,
            "test": case.test,
            "entry_point": case.entry_point,
        }
        work_directory = tempfile.TemporaryDirectory(prefix="cbv-bcb-work-")
        work_path = Path(work_directory.name).resolve()
        payload["sandbox_work_dir"] = str(work_path)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="cbv-bcb-",
                                             delete=False, encoding="utf-8")
        output_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".log", prefix="cbv-bcb-output-",
                                                    delete=False, encoding="utf-8")
        payload_path, output_path = Path(handle.name), Path(output_handle.name)
        try:
            json.dump(payload, handle)
            handle.close()
            os.chmod(payload_path, 0o600)
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    [sys.executable, "-I", str(self.worker_path), str(payload_path)],
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                    preexec_fn=self._set_limits,
                    env=self._clean_env(work_path),
                )
                output_handle.flush()
                latency = time.perf_counter() - started
                lines = [
                    line
                    for line in output_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if line.strip()
                ]
                body = json.loads(lines[-1]) if process.returncode == 0 and lines else {}
                passed = bool(body.get("passed", False))
                status = str(body.get("status", f"EVALUATOR_ERROR:worker_exit_{process.returncode}"))
                if status.startswith("EVALUATOR_ERROR") or status.startswith("sandbox_setup_failed"):
                    raise BigCodeBenchEvaluatorError(status)
            except subprocess.TimeoutExpired:
                latency, passed, status = time.perf_counter() - started, False, "TIMEOUT"
            return self._record(run_id, case.task_id, branch, code, passed, status, latency)
        finally:
            output_handle.close()
            payload_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            work_directory.cleanup()

    def _record(self, run_id: str, task_id: str, branch: str, code: str,
                passed: bool, status: str, latency: float) -> EvaluationResult:
        result = EvaluationResult(
            evaluation_id=content_id("eval", {"run_id": run_id, "task_id": task_id,
                                               "branch": branch, "code": code}),
            run_id=run_id, task_id=task_id, branch=branch, passed=passed, status=status,
            latency_seconds=latency, sandbox="bigcodebench-isolated-tests+landlock+seccomp+rlimits",
        )
        self.evaluation_log.append(result)
        return result

    @staticmethod
    def _clean_env(home: Path) -> dict[str, str]:
        scratch = str(home)
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": scratch,
            "TMPDIR": scratch,
            "TMP": scratch,
            "TEMP": scratch,
            "MPLCONFIGDIR": scratch,
            "XDG_CACHE_HOME": scratch,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
        }

    def _set_limits(self) -> None:
        memory = self.config.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = max(1, self.config.timeout_seconds - 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        # RLIMIT_NPROC is user-wide and counts threads, so a fixed value such as 1 or 64 can be
        # exhausted already on a shared host. Preserve defense in depth while granting this worker
        # only a small allowance above the user's current task count. Seccomp separately blocks
        # fork/vfork and process-form clone while allowing CLONE_THREAD.
        current_tasks = 0
        uid = os.getuid()
        for status_path in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = status_path.read_text(encoding="utf-8").splitlines()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            owner = next((line for line in fields if line.startswith("Uid:")), "")
            if not owner or int(owner.split()[1]) != uid:
                continue
            threads = next((line for line in fields if line.startswith("Threads:")), "Threads:\t1")
            current_tasks += int(threads.split()[1])
        worker_task_limit = current_tasks + max(16, self.config.max_processes)
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        if hard_limit != resource.RLIM_INFINITY:
            worker_task_limit = min(worker_task_limit, hard_limit)
        resource.setrlimit(resource.RLIMIT_NPROC, (worker_task_limit, worker_task_limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
