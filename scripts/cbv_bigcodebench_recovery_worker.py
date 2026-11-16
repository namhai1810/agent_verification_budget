#!/usr/bin/env python3
"""BigCodeBench-only recovery worker with a tamper-resistant result channel.

Some benchmark tests temporarily replace ``sys.stdout`` and can fail before restoring
it.  The normal worker then exits successfully while its final JSON is trapped in the
test's StringIO.  This recovery worker writes the protocol record to ``sys.__stdout__``.
It otherwise reuses the exact sandbox and BigCodeBench execution implementation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from agent_verify.evaluation import worker


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--probe-isolation":
        return worker.probe_isolation()
    started = time.perf_counter()
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            payload = json.load(handle)
        if str(payload.get("mode")) != "bigcodebench":
            raise RuntimeError("Recovery worker accepts BigCodeBench payloads only")
        writable_root = Path(str(payload["sandbox_work_dir"])).resolve(strict=True)
        if not writable_root.is_dir() or writable_root.is_symlink():
            raise RuntimeError("Invalid BigCodeBench sandbox_work_dir")
        os.chdir(writable_root)
        try:
            worker.restrict_filesystem(writable_root=writable_root)
            worker.restrict_syscalls()
        except Exception as exc:
            result = {
                "passed": False,
                "status": f"sandbox_setup_failed:{type(exc).__name__}:{exc}",
            }
        else:
            result = worker._bigcodebench_result(payload)
    except BaseException as exc:
        result = {"passed": False, "status": f"EVALUATOR_ERROR:{type(exc).__name__}"}
    result["worker_seconds"] = time.perf_counter() - started
    stream = sys.__stdout__
    if stream is None:
        raise RuntimeError("Original stdout is unavailable")
    stream.write(json.dumps(result, sort_keys=True) + "\n")
    stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
