import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def get_timestamp():
    t = time.localtime()
    date = f"{t.tm_year}{str(t.tm_mon).zfill(2)}{str(t.tm_mday).zfill(2)}"
    clock = (f"{str(t.tm_hour).zfill(2)}{str(t.tm_min).zfill(2)}"
             f"{str(t.tm_sec).zfill(2)}")
    timestamp = date + "_" + clock
    return timestamp

timestamp = get_timestamp()

