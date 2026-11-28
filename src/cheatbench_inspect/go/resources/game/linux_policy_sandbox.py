# Adapted from https://github.com/centerforaisafety/cheatbench/blob/96e05a6d61faf33be788f75dd01998066045ad02/tasks/go/linux_policy_sandbox.py
# Copyright (c) 2026 centerforaisafety and Long Phan. Licensed under MIT.

"""Trusted worker-only Linux confinement, installed before importing edited code."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from pathlib import Path
from typing import Any


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class Comparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def enforce(read_paths: list[str]) -> dict[str, Any]:
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Unsupported Linux architecture; no unsandboxed fallback")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    sec = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Comparison),
    ]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError(f"Landlock unavailable (errno={ctypes.get_errno()}); no fallback")
    if libc.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    attrs = Ruleset((1 << 13) - 1)
    fd = libc.syscall(444, ctypes.byref(attrs), ctypes.sizeof(attrs), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Landlock ruleset creation failed")
    try:
        for name in read_paths:
            path = Path(name)
            allowed = (1 << 2) | ((1 << 3) if path.is_dir() else 0)
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(allowed, path_fd)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), "Landlock rule insertion failed")
            finally:
                os.close(path_fd)
        if libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), "Landlock restriction failed")
    finally:
        os.close(fd)

    ctx = sec.seccomp_init(0x00050000 | errno.EPERM)  # SCMP_ACT_ERRNO
    if not ctx:
        raise RuntimeError("seccomp_init failed")

    def allow(name: str, comparisons: list[Comparison] | tuple[Comparison, ...] = ()) -> None:
        number = sec.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            return
        items = (Comparison * len(comparisons))(*comparisons)
        code = sec.seccomp_rule_add_array(ctx, 0x7FFF0000, number, len(items), items)
        if code:
            raise RuntimeError(f"seccomp rule {name} failed: {code}")

    try:
        for name in (
            "read",
            "pread64",
            "close",
            "fstat",
            "newfstatat",
            "stat",
            "lstat",
            "statx",
            "lseek",
            "getdents64",
            "readlink",
            "readlinkat",
            "access",
            "faccessat",
            "mmap",
            "mprotect",
            "munmap",
            "mremap",
            "brk",
            "madvise",
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "sigaltstack",
            "futex",
            "getrandom",
            "clock_gettime",
            "clock_getres",
            "gettimeofday",
            "time",
            "nanosleep",
            "clock_nanosleep",
            "sched_yield",
            "getpid",
            "getppid",
            "gettid",
            "getuid",
            "geteuid",
            "getgid",
            "getegid",
            "uname",
            "getcwd",
            "getrusage",
            "times",
            "exit",
            "exit_group",
        ):
            allow(name)
        for descriptor in (1, 2):
            allow("write", [Comparison(0, 4, descriptor, 0)])
        forbidden = os.O_ACCMODE | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        forbidden |= getattr(os, "O_TMPFILE", 0) & ~os.O_DIRECTORY
        allow("open", [Comparison(1, 7, forbidden, 0)])
        allow("openat", [Comparison(2, 7, forbidden, 0)])
        code = sec.seccomp_load(ctx)
        if code:
            raise RuntimeError(f"seccomp_load failed: {code}")
    finally:
        sec.seccomp_release(ctx)
    return {"backend": "linux-landlock-seccomp", "landlock_abi": abi}
