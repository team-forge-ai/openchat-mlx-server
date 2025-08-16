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
    """Delegate to local server CLI (forked from mlx_lm.server)."""
    try:
        from openchat_mlx_server.mlx_lm.server import main as mlx_main
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import local server: {exc}")
        return 1

    # mlx_lm.server.main() parses sys.argv itself and runs the HTTP server
    try:
        mlx_main()
    except KeyboardInterrupt:
        # Graceful shutdown on SIGINT without noisy traceback (important for PyInstaller)
        return 0
    except SystemExit as system_exit:
        # Respect explicit exit codes from upstream CLI
        try:
            return int(system_exit.code) if system_exit.code is not None else 0
        except Exception:
            return 0
    except Exception as unexpected_error:
        # Surface unexpected errors with a concise message and non-zero exit
        print(f"Unhandled exception: {unexpected_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())