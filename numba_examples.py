import numpy as np
import numba
from functools import wraps
from time import process_time

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(process_time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(process_time() * 1000)) - start
            print(
                f"Total execution time {func.__name__}: {end_ if end_ > 0 else 0} ms"
            )

    return _time_it

# np zeros_like is not accepted by numba
@measure
def search_min_np(grid):
    mins = np.zeros_like(grid, dtype=bool)
    for i in range(1, grid.shape[1] - 1):
        for j in range(1, grid.shape[0] - 1):
            if (grid[j, i] < grid[j - 1, i - 1] and
                    grid[j, i] < grid[j - 1, i] and
                    grid[j, i] < grid[j - 1, i + 1] and
                    grid[j, i] < grid[j, i - 1] and
                    grid[j, i] < grid[j, i + 1] and
                    grid[j, i] < grid[j + 1, i - 1] and
                    grid[j, i] < grid[j + 1, i] and
                    grid[j, i] < grid[j + 1, i + 1]):
                mins[i, j] = True
    return np.nonzero(mins)


@measure
def search_min_np_jit(grid):
    mins = np.zeros_like(grid, dtype=bool)
    _search_min(grid, mins)
    return np.nonzero(mins)


# Option accepted by numba the only thing is that we need to move the zeros_like outside the function
# Also the decorator is not allowed in terms of f strings and use of kwargs because are dictionaries
@numba.jit(nopython=True)
def _search_min(grid, mins):
    for i in range(1, grid.shape[1] - 1):
        for j in range(1, grid.shape[0] - 1):
            if (grid[j, i] < grid[j - 1, i - 1] and
                    grid[j, i] < grid[j - 1, i] and
                    grid[j, i] < grid[j - 1, i + 1] and
                    grid[j, i] < grid[j, i - 1] and
                    grid[j, i] < grid[j, i + 1] and
                    grid[j, i] < grid[j + 1, i - 1] and
                    grid[j, i] < grid[j + 1, i] and
                    grid[j, i] < grid[j + 1, i + 1]):
                mins[i, j] = True


if __name__ == '__main__':
    data = np.random.randn(2000, 2000)

    print("Eval function search_min_np")
    search_min_np(data)

    try:
        print("Eval function search_min_np with jit decorator")
        find_min_jit = numba.jit(nopython=True)(search_min_np)
        find_min_jit(data)
    except Exception as e:
        print(f"Failed executing find_min_jit. Reason {str(e)}")

    search_min_np_jit(data)