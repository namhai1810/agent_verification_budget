"""Public-feedback-only sandbox worker for the execution repair pilot.

The worker never receives benchmark hidden tests. Hidden evaluation continues to use
the repository's original HumanEval/LiveCodeBench/BigCodeBench runners.
"""

from __future__ import annotations

import doctest
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from agent_verify.evaluation.worker import restrict_filesystem, restrict_syscalls


def _clip(value: object, limit: int = 6000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _humaneval(payload: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(str(payload["code"]), "<public-humaneval-code>", "exec"), namespace, namespace)
    entry_point = str(payload.get("entry_point", ""))
    if entry_point and not callable(namespace.get(entry_point)):
        return {
            "available": True,
            "passed": False,
            "status": "ENTRY_POINT_MISSING",
            "feedback": f"Public smoke check: required callable {entry_point!r} is missing.",
            "test_count": 1,
        }
    parser = doctest.DocTestParser()
    test = parser.get_doctest(
        str(payload.get("prompt", "")), namespace, str(payload.get("task_id", "task")),
        "<public-humaneval-prompt>", 0,
    )
    if not test.examples:
        return {
            "available": False,
            "passed": True,
            "status": "NO_PUBLIC_EXAMPLES",
            "feedback": (
                "No executable public examples were present. Syntax/import and required-entry-point "
                "checks passed; no semantic execution feedback is available."
            ),
            "test_count": 0,
        }
    output: list[str] = []
    runner = doctest.DocTestRunner(verbose=False)
    result = runner.run(test, out=output.append, clear_globs=False)
    passed = result.failed == 0
    feedback = "".join(output).strip()
    if passed:
        feedback = f"All {result.attempted} public doctest example(s) passed."
    return {
        "available": True,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "feedback": _clip(feedback),
        "test_count": result.attempted,
        "failed_count": result.failed,
    }


def _livecodebench(payload: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(Path(payload["official_repo"]).resolve()))
    from lcb_runner.evaluation.testing_util import run_test

    tests = list(payload.get("public_tests", []))
    if not tests:
        return {
            "available": False,
            "passed": True,
            "status": "NO_PUBLIC_TESTS",
            "feedback": "No public LiveCodeBench tests were available.",
            "test_count": 0,
        }
    sample = {
        "input_output": json.dumps(
            {
                "inputs": [item.get("input") for item in tests],
                "outputs": [item.get("output") for item in tests],
                "fn_name": payload.get("fn_name"),
            },
            ensure_ascii=False,
        )
    }
    results, metadata = run_test(
        sample,
        test=str(payload["code"]),
        debug=False,
        timeout=int(payload.get("per_test_timeout", 6)),
    )
    normalized = [bool(value) if type(value).__name__ == "bool_" else value for value in results]
    failed = [index for index, value in enumerate(normalized) if value is not True]
    details: list[str] = []
    for index in failed[:5]:
        case = tests[index] if index < len(tests) else {}
        details.append(
            f"case {index}: input={case.get('input')!r}; expected={case.get('output')!r}; "
            f"result_code={normalized[index]!r}"
        )
    if failed:
        details.append("Public evaluator metadata: " + _clip(metadata, 2500))
    passed = bool(normalized) and not failed
    return {
        "available": True,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "feedback": (
            f"All {len(tests)} public test(s) passed."
            if passed
            else _clip("Public test failures:\n" + "\n".join(details))
        ),
        "test_count": len(tests),
        "failed_count": len(failed),
    }


def _bigcodebench(payload: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(str(payload["code"]), "<public-bigcodebench-code>", "exec"), namespace, namespace)
    entry_point = str(payload.get("entry_point", ""))
    passed = bool(entry_point and callable(namespace.get(entry_point)))
    return {
        "available": False,
        "passed": passed,
        "status": "NO_PUBLIC_TESTS" if passed else "ENTRY_POINT_MISSING",
        "feedback": (
            "BigCodeBench exposes no public tests for this task. Syntax/import and required-entry-"
            "point checks passed; no semantic execution feedback is available."
            if passed
            else f"Public smoke check: required callable {entry_point!r} is missing."
        ),
        "test_count": 0,
    }


def main() -> int:
    started = time.perf_counter()
    try:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        dataset = str(payload["dataset"])
        writable_root = Path(str(payload["sandbox_work_dir"])).resolve(strict=True)
        if not writable_root.is_dir() or writable_root.is_symlink():
            raise ValueError("sandbox_work_dir must be a real directory")
        if dataset == "livecodebench":
            # Import the trusted official evaluator before Landlock removes repository access.
            official_repo = str(Path(payload["official_repo"]).resolve())
            sys.path.insert(0, official_repo)
            import lcb_runner.evaluation.testing_util  # noqa: F401
        os.chdir(writable_root)
        restrict_filesystem(writable_root=writable_root)
        restrict_syscalls()
        if dataset == "humaneval":
            result = _humaneval(payload)
        elif dataset == "livecodebench":
            result = _livecodebench(payload)
        elif dataset == "bigcodebench":
            result = _bigcodebench(payload)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    except BaseException as exc:  # Generated code is deliberately untrusted.
        result = {
            "available": True,
            "passed": False,
            "status": f"RUNTIME_ERROR:{type(exc).__name__}",
            "feedback": _clip(
                f"Public execution raised {type(exc).__name__}: {exc}\n"
                + "".join(traceback.format_exception_only(type(exc), exc))
            ),
            "test_count": 0,
        }
    result["latency_seconds"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
