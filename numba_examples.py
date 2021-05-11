import numpy as np
from functools import wraps
from time import process_time
import timeit
from numba import njit, jit


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


def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# print(f"Running fibonacci. Took {timeit.timeit('fibonacci(35)', 'from __main__ import fibonacci', number=30)}")


@jit(fastmath=True, cache=True, nopython=True)
def fibonacci_numba(n):
    if n < 2:
        return n
    else:
        return fibonacci_numba(n - 1) + fibonacci_numba(n - 2)

# print(f"Running fibonacci numba. Took "
#       f"{timeit.timeit('fibonacci_numba(35)', 'from __main__ import fibonacci_numba', number=30)}")


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


@jit(nopython=True)
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


# Option accepted by numba the only thing is that we need to move the zeros_like outside the function
# Also the decorator is not allowed in terms of f strings and use of kwargs because are dictionaries


if __name__ == '__main__':
    # data = np.random.randn(2000, 2000)

    #     SETUP_CODE = '''
    # from __main__ import search_min_np
    # import numpy as np'''
    #
    #     TEST_CODE = '''
    # data = np.random.randn(2000, 2000)'''
    #
    #     print("Eval function search_min_np")
    #     print(timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, number=30))
    #     # measure(search_min_np(data))
    #
    #     SETUP_CODE = '''
    # import numpy as np
    # from numba import njit, jit
    # from __main__ import search_min_np
    # jit(nopython=True)(search_min_np)
    # '''
    #
    #     TEST_CODE = '''
    # data = np.random.randn(2000, 2000)'''
    #
    #     try:
    #         print("Eval function search_min_np with jit decorator")
    #         find_min_jit = jit(nopython=True)(search_min_np)
    #         # print(timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, number=30))
    #         find_min_jit(data)
    #     except Exception as e:
    #         print(f"Failed executing find_min_jit. Reason {str(e)}")
    #
    #     print("Eval function search_min_np_jit")
    #     search_min_np_jit(data)

    print("Running fibonacci")
    print(timeit.timeit("fibonacci(35)", "from __main__ import fibonacci", number=30))
    print("Running fibonacci numba")
    print(timeit.timeit("fibonacci_numba(35)", "from __main__ import fibonacci_numba", number=30))
