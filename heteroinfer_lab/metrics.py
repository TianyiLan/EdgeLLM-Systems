"""Metric aggregation helpers."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable


def mean_metric(records: Iterable[dict], key: str, digits: int = 2) -> float:
    """Return rounded mean for a numeric metric key."""
    values = [float(record[key]) for record in records]
    return round(mean(values), digits)


def std_metric(records: Iterable[dict], key: str, digits: int = 2) -> float:
    """Return rounded population standard deviation for a numeric metric key."""
    values = [float(record[key]) for record in records]
    return round(pstdev(values), digits) if len(values) > 1 else 0.0


def tokens_per_second(tpot_ms: float) -> float:
    """Convert TPOT in milliseconds to tokens per second."""
    return 1000.0 / tpot_ms
