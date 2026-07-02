"""
Logging setup shared by the entrypoints.

- Root logger -> stderr (so tqdm bars on stderr and logs interleave cleanly via
  logging_redirect_tqdm) plus an optional FileHandler in the run dir.
- Line-buffered stdout so `tail -f` works under nohup.
- A tqdm factory that auto-disables under a non-TTY (redirected logs stay clean).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm as _tqdm


class _TqdmLoggingHandler(logging.Handler):
    """Emit log records via tqdm.write so active progress bars are not clobbered."""

    def emit(self, record):
        try:
            _tqdm.write(self.format(record), file=sys.stderr)
            self.flush()
        except Exception:  # pragma: no cover
            self.handleError(record)


def line_buffer_stdout() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


def setup_logging(run_dir: Optional[str] = None, level: str = "INFO",
                  filename: str = "run.log") -> logging.Logger:
    line_buffer_stdout()
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = _TqdmLoggingHandler()  # bars + logs coexist (tqdm.write to stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(run_dir) / filename)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


def is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def pbar(iterable=None, disable_when_not_tty: bool = True, **kwargs):
    """tqdm wrapper: dynamic width, auto-disable under non-TTY."""
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    if disable_when_not_tty and not is_tty():
        kwargs["disable"] = True
    return _tqdm(iterable, **kwargs)
