import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


def moore_penrose(A, delta=1, e=1e-10):
    A = np.array(A, dtype=float)
    aT = A.T
    aaT = A @ aT
    E = np.eye(A.shape[0])
    prev = aT @ np.linalg.inv(aaT + delta**2 * E)
    while True:
        delta = delta / 2
        cur = aT @ np.linalg.inv(aaT + delta**2 * E)
        if np.linalg.norm(cur - prev) < e:
            break
        prev = cur
    return cur


def greville(M):
    M = np.array(M, dtype=float)
    ai = M[0:1]
    if np.count_nonzero(ai[0]) == 0:
        res = np.zeros_like(ai.T)
    else:
        res = ai.T/(ai @ ai.T)

    n = M.shape[0]
    for i in range(1, n):
        z_a = np.eye(res.shape[0]) - (res @ M[:i])
        r_a = res @ res.T
        ai = M[i:i+1]

        condition = (ai @ z_a) @ ai.T
        if np.count_nonzero(condition) != 1:
            a_inv = (r_a @ ai.T) / (1 + (ai @ r_a) @ ai.T)
        else:
            a_inv = (z_a @ ai.T) / condition

        res -= a_inv @ (ai @ res)
        res = np.concatenate((res, a_inv), axis=1)
    return res


if __name__ == "__main__":
    X, Y = image.imread('x1.bmp'), image.imread('y6.bmp')
    plt.imshow(X, cmap='gray')
    plt.show()

    A_moore = Y @ moore_penrose(X)
    A_greville = Y @ greville(X)

    plt.imshow(Y, cmap='gray')
    plt.show()
    plt.imshow(A_moore @ X, cmap='gray')
    plt.show()
    plt.imshow(A_greville @ X, cmap='gray')
    plt.show()
