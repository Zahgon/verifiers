import asyncio
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

THREAD_LOCAL_STORAGE = threading.local()


def get_thread_local_storage() -> threading.local:
    """Get the thread-local storage for the current thread."""
    pass


def get_or_create_thread_attr(
    key: str, factory: Callable[..., Any], *args, **kwargs
) -> Any:
    """Get value from thread-local storage, creating it if it doesn't exist."""
    pass


def get_or_create_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create event loop for current thread. Reuses loop to avoid closing it."""
    pass


# --- Executor registry & scaling ---

# Default scaling: 1:1 concurrency to max_workers
ScalingFn = Callable[[int], int]


def _default_scaling(concurrency: int) -> int:
    pass


Executor = ThreadPoolExecutor | ProcessPoolExecutor
_executor_registry: dict[str, tuple[Executor, ScalingFn]] = {}
_default_executor: ThreadPoolExecutor | None = None
_target_concurrency: int | None = None  # sticky target from last scale_executors call


def _resize(executor: Executor, max_workers: int) -> None:
    """Resize an executor in-place. Workers are spawned lazily so
    raising the limit simply allows more workers on the next submit."""
    pass


def register_executor(
    name: str,
    executor: Executor,
    scaling_fn: ScalingFn | None = None,
) -> None:
    """Register an executor so it is resized by future :func:`scale_executors` calls.

    *scaling_fn* maps concurrency → max_workers for this executor.
    Defaults to 1:1 if not provided.

    If :func:`scale_executors` was already called, the executor is immediately
    resized using the scaling function.
    """
    pass


def unregister_executor(name: str) -> None:
    """Remove a previously registered executor (does **not** shut it down)."""
    _executor_registry.pop(name, None)


def scale_executors(concurrency: int) -> int:
    """Scale the default event-loop executor **and** all registered executors.

    Each registered executor applies its own scaling function to map
    *concurrency* to a max_workers value (default 1:1).

    If a running event loop exists, the default executor is bound to it
    immediately.  Otherwise the executor is only created/resized and the
    caller must call :func:`install_default_executor` once inside the real
    loop (e.g. at the start of ``async def run()``).

    Returns *concurrency*.
    """
    pass


def install_default_executor() -> None:
    """Bind the default executor to the **currently running** event loop.

    Call this early inside an ``async`` function (after ``asyncio.run()`` has
    created the real loop) so that ``run_in_executor(None, ...)`` uses the
    scaled thread pool.  Safe to call multiple times — it is a no-op if no
    default executor has been created yet.
    """
    if _default_executor is not None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(_default_executor)
        logger.debug(
            f"Installed default executor (max_workers={_default_executor._max_workers}) "
            f"on loop {id(loop)}"
        )


def shutdown_executors() -> None:
    """Shut down the default executor and all registered executors."""
    pass
