import numpy as np
from typing import Optional
from numba import njit


@njit
def c(x_i: float, x_i_1: float, y_j: float, constant: float) -> float:
    if (x_i_1 <= x_i and x_i <= y_j) or (x_i_1 >= x_i and x_i >= y_j):
        return constant
    return float(constant + min(np.abs(x_i - x_i_1), np.abs(x_i - y_j)))


@njit
def move_split_merge(x: np.ndarray, y: np.ndarray, constant: Optional[float] = 0.5) -> float:
    constant = constant or 0.5
    m = x.shape[0]
    n = y.shape[0]

    cost = np.zeros((m, n), dtype=np.float_)
    cost[0, 0] = np.abs(x[0] - y[0])
    for i in range(1, m):
        cost[i, 0] = cost[i - 1, 0] + c(x[i], x[i - 1], y[0], constant)

    for j in range(1, n):
        cost[0, j] = cost[0, j - 1] + c(y[j], x[0], y[j - 1], constant)

    for i in range(1, m):
        for j in range(1, n):
            cost[i, j] = min(
                cost[i - 1, j - 1] + np.abs(x[i] - y[j]),
                cost[i - 1, j] + c(x[i], x[i - 1], y[j], constant),
                cost[i, j - 1] + c(y[j], x[i], y[j - 1], constant),
            )

    return float(cost[m - 1, n - 1])


if __name__ == "__main__":
    x = np.random.rand(100)
    y = np.random.rand(115)
    print("x", "[" + ", ".join(map(str, x)) + "]")
    print("y", "[" + ", ".join(map(str, y)) + "]")
    print("msm", move_split_merge(x, y, 0.5))
