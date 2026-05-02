from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Apply a consistent log format across all modules."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
