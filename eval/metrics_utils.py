# eval/metrics_utils.py

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple


def time_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """
    Measure wall-clock time for a function call in seconds.
    Returns (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, (end - start)


def dummy_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tiny helper for the interactive CLI demo.
    """
    return {
        "num_results": len(results),
    }
