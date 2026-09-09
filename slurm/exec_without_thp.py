"""Disable transparent huge pages for this process, then exec a command."""

import ctypes
import os
import sys


if len(sys.argv) < 2:
    raise SystemExit("Usage: exec_without_thp.py COMMAND [ARGS ...]")

libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
libc.prctl.restype = ctypes.c_int

# Linux prctl.h: the setting survives fork and exec. No system-wide changes.
PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42
if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) == -1:
    err = ctypes.get_errno()
    raise OSError(err, os.strerror(err))
if libc.prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0) != 1:
    raise RuntimeError("Failed to verify process-level THP disable")

print("THP_DIAG: PR_GET_THP_DISABLE=1; inherited by training and data-loader workers", flush=True)
os.execvp(sys.argv[1], sys.argv[1:])
