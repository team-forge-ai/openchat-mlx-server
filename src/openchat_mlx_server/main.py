#!/usr/bin/env python3
"""Thin wrapper around upstream mlx_lm.server.

This module delegates to mlx_lm.server so we don't maintain a separate server stack.
Use it exactly like `python -m mlx_lm server ...` or `mlx_lm.server ...`.
"""

from __future__ import annotations

import sys


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