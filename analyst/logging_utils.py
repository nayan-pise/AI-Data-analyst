"""Structured JSON logging plus lightweight in-process metrics (observability)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

_LOGGER_NAME = "analyst"
_lock = threading.Lock()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def get_logger(log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = _JsonFormatter()
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    directory = log_dir or os.getenv("ANALYST_LOG_DIR", "logs")
    try:
        os.makedirs(directory, exist_ok=True)
        fh = logging.FileHandler(os.path.join(directory, "analyst.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:  # read-only filesystem etc. -> console logging only
        pass
    return logger


class Metrics:
    """Tiny thread-safe counter/latency registry shown in the UI sidebar."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.latency_total: dict[str, float] = defaultdict(float)
        self.errors: dict[str, int] = defaultdict(int)

    def record(self, name: str, seconds: float, ok: bool = True) -> None:
        with _lock:
            self.counts[name] += 1
            self.latency_total[name] += seconds
            if not ok:
                self.errors[name] += 1

    def snapshot(self) -> list[dict]:
        with _lock:
            return [
                {
                    "metric": k,
                    "calls": self.counts[k],
                    "errors": self.errors[k],
                    "avg_ms": round(1000 * self.latency_total[k] / max(self.counts[k], 1), 1),
                }
                for k in sorted(self.counts)
            ]


METRICS = Metrics()


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    get_logger().log(level, event, extra={"fields": fields})


@contextmanager
def timed(name: str, **fields):
    """Time a block, record the metric and emit a log line (also on failure)."""
    start = time.perf_counter()
    ok = True
    try:
        yield
    except Exception as exc:
        ok = False
        fields["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - start
        METRICS.record(name, elapsed, ok)
        log_event(name, level=logging.INFO if ok else logging.WARNING,
                  duration_ms=round(elapsed * 1000, 1), ok=ok, **fields)
