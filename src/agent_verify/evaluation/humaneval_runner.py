from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_verify.config import EvaluationConfig
from agent_verify.logging import JsonlStore, content_id
from agent_verify.schemas import EvaluationResult, PrivateEvaluationCase


class SandboxUnavailable(RuntimeError):
    pass


class HumanEvalRunner:
    def __init__(
        self,
        config: EvaluationConfig,
        evaluation_log: JsonlStore,
    ) -> None:
        self.config = config
        self.evaluation_log = evaluation_log
        self.worker_path = Path(__file__).with_name("worker.py").resolve()
        self._sandbox_checked = False

    def check_sandbox(self) -> None:
        if self._sandbox_checked:
            return
        probe = subprocess.run(
            [
                sys.executable,
                "-I",
                str(self.worker_path),
                "--probe-isolation",
            ],
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
        case: PrivateEvaluationCase,
        code: str,
        branch: str,
    ) -> EvaluationResult:
        self.check_sandbox()
        payload = {"code": code, "test": case.test, "entry_point": case.entry_point}
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="agent-verify-", delete=False, encoding="utf-8"
        )
        payload_path = Path(handle.name)
        output_handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="agent-verify-output-", delete=False, encoding="utf-8"
        )
        output_path = Path(output_handle.name)
        try:
            json.dump(payload, handle)
            handle.close()
            os.chmod(payload_path, 0o600)
            command = [
                sys.executable,
                "-I",
                str(self.worker_path),
                str(payload_path),
            ]
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    command,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                    preexec_fn=self._set_limits,
                    env=self._clean_env(payload_path.parent),
                )
                output_handle.flush()
                latency = time.perf_counter() - started
                if process.returncode != 0:
                    passed = False
                    status = f"sandbox_exit:{process.returncode}"
                else:
                    lines = [
                        line
                        for line in output_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip()
                    ]
                    worker_result = json.loads(lines[-1]) if lines else {}
                    passed = bool(worker_result.get("passed", False))
                    status = str(worker_result.get("status", "invalid_worker_output"))
            except subprocess.TimeoutExpired:
                latency = time.perf_counter() - started
                passed = False
                status = "timeout"

            identity = {
                "run_id": run_id,
                "task_id": case.task_id,
                "branch": branch,
                "code": code,
            }
            result = EvaluationResult(
                evaluation_id=content_id("eval", identity),
                run_id=run_id,
                task_id=case.task_id,
                branch=branch,
                passed=passed,
                status=status,
                latency_seconds=latency,
                sandbox="landlock-read-allowlist+seccomp-network-process-deny+rlimits",
            )
            self.evaluation_log.append(result)
            return result
        finally:
            output_handle.close()
            try:
                payload_path.unlink()
            except FileNotFoundError:
                pass
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _clean_env(home: Path) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
        }

    def _set_limits(self) -> None:
        memory_bytes = self.config.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_seconds = max(1, self.config.timeout_seconds - 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(
            resource.RLIMIT_NPROC,
            (self.config.max_processes, self.config.max_processes),
        )
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
