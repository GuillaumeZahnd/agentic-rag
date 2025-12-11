import time
import torch
import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar, Awaitable


# Type aliases
T = TypeVar('T')
SynchFunc = Callable[..., T]
AsyncFunc = Callable[..., Awaitable[T]]


def sync_timer(func: SynchFunc) -> SynchFunc:
    """
    Decorator that measures the execution time of a synchronous function, while ensuring correct CUDA synchronization.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        is_cuda_available = hasattr(torch, 'cuda') and torch.cuda.is_available()

        if is_cuda_available:
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)

        if is_cuda_available:
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        runtime = end_time - start_time

        print("⏳ {}: {:.2f}s".format(func.__name__, runtime))
        return result

    return wrapper


def async_timer(func: AsyncFunc) -> AsyncFunc:
    """
    Decorator that measures the execution time of an asynchronous function, while ensuring correct CUDA synchronization.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):

        is_cuda_available = hasattr(torch, 'cuda') and torch.cuda.is_available()

        if is_cuda_available:
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = await func(*args, **kwargs)

        if is_cuda_available:
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        runtime = end_time - start_time

        print("⏳ {}: {:.2f}s".format(func.__name__, runtime))
        return result

    return wrapper
