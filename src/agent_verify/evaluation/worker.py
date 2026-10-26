"""Minimal worker executed only inside the OS sandbox.

This module intentionally has no imports from the orchestration package.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import os
import socket
import sys
import time
from pathlib import Path

# The soname fallback avoids a subprocess-based ldconfig lookup after resource limits.
SECCOMP_LIBRARY = ctypes.util.find_library("seccomp") or "libseccomp.so.2"


class LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


def restrict_filesystem() -> None:
    """Allow only read/execute under system and active Python runtime directories."""
    libc = ctypes.CDLL(None, use_errno=True)
    create_ruleset, add_rule, restrict_self = (444, 445, 446)
    abi = libc.syscall(create_ruleset, 0, 0, 1)
    if abi < 1:
        raise RuntimeError(f"Landlock unavailable (errno={ctypes.get_errno()})")

    execute = 1 << 0
    read_file = 1 << 2
    read_dir = 1 << 3
    handled = sum(1 << bit for bit in range(13))
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14
    if abi >= 5:
        handled |= 1 << 15
    ruleset_attr = LandlockRulesetAttr(handled_access_fs=handled)
    ruleset_fd = libc.syscall(
        create_ruleset,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        raise RuntimeError(f"landlock_create_ruleset failed (errno={ctypes.get_errno()})")

    read_execute = execute | read_file | read_dir
    allowed_roots = [Path(sys.prefix), Path("/usr"), Path("/lib"), Path("/lib64"), Path("/bin")]
    try:
        for path in allowed_roots:
            if not path.exists():
                continue
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = LandlockPathBeneathAttr(
                    allowed_access=read_execute & handled,
                    parent_fd=path_fd,
                )
                if libc.syscall(add_rule, ruleset_fd, 1, ctypes.byref(rule), 0) != 0:
                    raise RuntimeError(f"landlock_add_rule failed for {path} (errno={ctypes.get_errno()})")
            finally:
                os.close(path_fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise RuntimeError(f"prctl(NO_NEW_PRIVS) failed (errno={ctypes.get_errno()})")
        if libc.syscall(restrict_self, ruleset_fd, 0) != 0:
            raise RuntimeError(f"landlock_restrict_self failed (errno={ctypes.get_errno()})")
    finally:
        os.close(ruleset_fd)


def restrict_syscalls() -> None:
    """Install an irreversible seccomp filter for network, process, and escape syscalls."""
    if not SECCOMP_LIBRARY:
        raise RuntimeError("libseccomp is unavailable")
    library = ctypes.CDLL(SECCOMP_LIBRARY, use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]

    allow = 0x7FFF0000
    deny_eperm = 0x00050000 | errno.EPERM
    context = library.seccomp_init(allow)
    if not context:
        raise RuntimeError("seccomp_init failed")
    blocked = (
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "recvfrom",
        "sendmsg",
        "recvmsg",
        "recvmmsg",
        "sendmmsg",
        "getsockname",
        "getpeername",
        "setsockopt",
        "getsockopt",
        "shutdown",
        "clone",
        "clone3",
        "fork",
        "vfork",
        "execve",
        "execveat",
        "ptrace",
        "mount",
        "umount2",
        "pivot_root",
        "chroot",
        "unshare",
        "setns",
        "bpf",
        "perf_event_open",
        "userfaultfd",
        "keyctl",
        "add_key",
        "request_key",
    )
    try:
        for name in blocked:
            syscall = library.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and library.seccomp_rule_add(context, deny_eperm, syscall, 0) != 0:
                raise RuntimeError(f"Could not block syscall: {name}")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        library.seccomp_release(context)


def probe_isolation() -> int:
    restrict_filesystem()
    restrict_syscalls()
    checks: dict[str, bool] = {}
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        checks["network_blocked"] = True
    else:
        checks["network_blocked"] = False
    try:
        open("/etc/passwd", encoding="utf-8").read(1)
    except PermissionError:
        checks["sensitive_read_blocked"] = True
    else:
        checks["sensitive_read_blocked"] = False
    try:
        open("/tmp/agent-verify-probe", "w", encoding="utf-8").write("x")
    except PermissionError:
        checks["filesystem_write_blocked"] = True
    else:
        checks["filesystem_write_blocked"] = False
    try:
        os.fork()
    except PermissionError:
        checks["process_creation_blocked"] = True
    else:
        checks["process_creation_blocked"] = False
    checks["isolation_ok"] = all(checks.values())
    print(json.dumps(checks, sort_keys=True), flush=True)
    return 0 if checks["isolation_ok"] else 1


def _livecodebench_result(run_test, payload: dict) -> dict:
    sample = {"input_output": payload["input_output"]}
    results, metadata = run_test(
        sample,
        test=payload["code"],
        debug=False,
        timeout=int(payload["per_test_timeout"]),
    )
    normalized = [bool(value) if type(value).__name__ == "bool_" else value for value in results]
    if normalized and all(value is True for value in normalized):
        return {"passed": True, "status": "PASS", "official_metadata": metadata}
    error_code = int(metadata.get("error_code", 0) or 0)
    if error_code == -3 or -3 in normalized:
        status = "TIMEOUT"
    elif error_code == -4 or -4 in normalized:
        status = "RUNTIME_ERROR"
    elif error_code in {-1, -2} or any(value in {-1, -2, False} for value in normalized):
        status = "FAIL"
    else:
        status = "EVALUATOR_ERROR"
    return {"passed": False, "status": status, "official_metadata": metadata}


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--probe-isolation":
        return probe_isolation()
    started = time.perf_counter()
    mode = "humaneval"
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            payload = json.load(handle)
        mode = str(payload.get("mode", "humaneval"))
        run_test = None
        if mode == "livecodebench":
            # Import trusted upstream code before Landlock removes repository access.
            sys.path.insert(0, str(Path(payload["official_repo"]).resolve()))
            from lcb_runner.evaluation.testing_util import run_test as official_run_test

            run_test = official_run_test
        try:
            restrict_filesystem()
            restrict_syscalls()
        except Exception as exc:
            result = {
                "passed": False,
                "status": f"sandbox_setup_failed:{type(exc).__name__}:{exc}",
                "worker_seconds": time.perf_counter() - started,
            }
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0

        if mode == "livecodebench":
            if run_test is None:
                raise RuntimeError("Official LiveCodeBench evaluator was not loaded")
            result = _livecodebench_result(run_test, payload)
        else:
            namespace: dict = {}
            source = payload["code"] + "\n" + payload["test"] + "\ncheck(" + payload["entry_point"] + ")\n"
            exec(compile(source, "<isolated-humaneval>", "exec"), namespace, namespace)
            result = {"passed": True, "status": "passed"}
    except BaseException as exc:  # The sandbox must contain arbitrary generated code.
        prefix = "EVALUATOR_ERROR" if mode == "livecodebench" else "failed"
        result = {"passed": False, "status": f"{prefix}:{type(exc).__name__}"}
    result["worker_seconds"] = time.perf_counter() - started
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
