#!/usr/bin/env python3
"""Thin wrapper around upstream mlx_lm.server.

This module delegates to mlx_lm.server so we don't maintain a separate server stack.
Use it exactly like `python -m mlx_lm server ...` or `mlx_lm.server ...`.
"""

from __future__ import annotations

import sys
import os
import shutil
import platform

# Mitigate PyInstaller + multiprocessing issue where child processes are spawned
# using this binary as the interpreter and pass CPython flags (-OO -B -S -I -c ...).
# We redirect multiprocessing to a real Python interpreter and prefer 'fork' on macOS.
try:
    import multiprocessing as _mp  # noqa: WPS433

    real_python = shutil.which("python3") or sys.executable
    if real_python:
        try:
            _mp.set_executable(real_python)
        except Exception:
            pass

    if platform.system() == "Darwin":
        try:
            _mp.set_start_method("fork", force=False)
        except RuntimeError:
            # Already set in this process; ignore
            pass
except Exception:
    # If multiprocessing is unavailable or configuration fails, proceed anyway
    pass


def main() -> int:
    """Delegate to upstream mlx_lm.server CLI."""
    try:
        from mlx_lm.server import main as mlx_main
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import mlx_lm.server: {exc}")
        return 1

    # mlx_lm.server.main() parses sys.argv itself and runs the HTTP server
    mlx_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())