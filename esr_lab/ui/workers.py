"""Async worker utilities for GUI tasks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class AsyncTaskRunner:
    """Small wrapper around ThreadPoolExecutor for GUI background jobs."""

    def __init__(self, *, max_workers: int = 2, thread_name_prefix: str = "simpleesr") -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit a callable to run in the background."""

        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """Shutdown worker threads."""

        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

