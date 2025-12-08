import time
import torch
from functools import wraps


def timer(func):
    """
    Decorator that measures the execution time of a synchronous function, while ensuring correct CUDA synchronization.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print("⏳ {}: {:.2f}s".format(func.__name__, runtime))
        return result

    return wrapper

