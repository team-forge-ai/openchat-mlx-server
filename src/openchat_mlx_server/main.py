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

def _install_safe_add_docstring() -> None:
    """Install a safe builtins.add_docstring to avoid numpy doc issues."""
    try:
        import builtins

        def safe_add_docstring(obj, doc):
            if doc is None:
                return
            if not isinstance(doc, str):
                try:
                    doc = str(doc)
                except Exception:
                    return
            try:
                obj.__doc__ = doc
            except Exception:
                pass

        builtins.add_docstring = safe_add_docstring
        try:
            import numpy  # noqa: F401
        except Exception:
            # If NumPy import fails here, continue; the shim remains installed
            pass
    except Exception:
        # Best effort. If we cannot install, proceed.
        pass


def _configure_multiprocessing_runtime() -> None:
    """Point multiprocessing helpers to a real Python interpreter.

    When this app is packaged as a single binary, Python's multiprocessing can
    attempt to spawn helper processes (like the resource tracker) by invoking
    the binary with interpreter flags (e.g. "-OO -B -S -I -c ..."), which our
    CLI cannot parse. We avoid that by telling multiprocessing which Python
    interpreter to use for child processes.

    Precedence:
      1) Environment variable `OPENCHAT_MLX_SERVER_PYTHON`
      2) A `python3` or `python` binary located next to this executable
      3) `python3` discovered on PATH
      4) Fallback to `sys.executable`
    """
    try:
        import multiprocessing as _mp  # noqa: WPS433
        from pathlib import Path

        configured = os.environ.get("OPENCHAT_MLX_SERVER_PYTHON")

        if not configured:
            try:
                exec_dir = Path(sys.executable).parent
                for name in ("python3", "python"):
                    candidate = exec_dir / name
                    if candidate.exists() and candidate.is_file():
                        configured = str(candidate)
                        break
            except Exception:
                configured = None

        if not configured:
            configured = shutil.which("python3")

        if not configured:
            configured = sys.executable

        try:
            _mp.set_executable(configured)
        except Exception:
            pass

        # Do not force 'fork' to keep PID semantics predictable for sidecar
        # supervisors. Ensure a context exists; default on macOS is 'spawn'.
        if platform.system() == "Darwin":
            try:
                _mp.get_start_method()
            except Exception as e:
                print(f"Failed to get multiprocessing start method: {e}", file=sys.stderr)
                _mp.set_start_method("spawn", force=False)
    except Exception:
        # Best-effort only; if we cannot configure, proceed with defaults.
        pass


def _set_own_process_group_if_possible() -> None:
    """Ensure this process is leader of its own process group on POSIX.

    This allows callers to send signals to the entire server tree using
    negative PIDs (e.g., kill(-pgid, SIGTERM)) for reliable shutdowns
    when using multiprocessing with 'fork'.
    """
    try:
        if os.name == "posix":  # Only available/meaningful on POSIX systems
            os.setpgrp()
    except Exception:
        # Best-effort only; ignore if unsupported or if permissions disallow
        pass


def main() -> int:
    """Delegate to local server CLI (forked from mlx_lm.server)."""
    # Place the server in its own process group so group-targeted signals
    # reach all descendants (important for reliable shutdown).
    _set_own_process_group_if_possible()
    _install_safe_add_docstring()
    _configure_multiprocessing_runtime()
    try:
        from openchat_mlx_server.mlx_lm.server import main as mlx_main
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import local server: {exc}", file=sys.stderr)
        return 1

    # Print current pid
    print(f"Current pid: {os.getpid()}", file=sys.stderr)

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