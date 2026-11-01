"""Timestamped logging to the terminal and to a per-run log file, written in real time."""

import logging
import os
import sys


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Logger that prints `YYYY-MM-DD HH:MM:SS level name: message` to stdout and `log_file`."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    # Replace handlers so repeated calls (e.g. one run after another) do not duplicate lines.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        # FileHandler flushes after every record, so the file is readable while a run is going.
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
