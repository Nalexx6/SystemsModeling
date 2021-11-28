import numpy as np

with open('y6.txt') as file:
    data = np.array([line.split() for line in file.readlines()], float).T

c3 = 0.2
c4 = 0.12
m1 = 12
m3 = 18
# c1, c2, m2 are unknown


def get_sensitivity_matrix(b):
    return np.array([
        [0, 1, 0, 0, 0, 0],
        [-(b[1] + b[0]) / m1, 0, b[1] / m1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [b[1] / b[2], 0, -(b[1] + c3) / b[2], 0, c3 / b[2], 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, c3 / m3, 0, -(c4 + c3) / m3, 0]
    ])


def calculate_model_derivatives(y, b):
    db0 = np.array([
        [0, 0, 0, 0, 0, 0],
        [- 1 / m1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    db1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [- 1 / m1, 0, 1 / m1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1 / b[2], 0, -1 / b[2], 0, c3 / b[2], 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    db2 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [- b[1] / (b[2] ** 2), 0, (b[1] + c3) / (b[2] ** 2), 0, -c3 / (b[2] ** 2), 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    db0 = db0 @ y
    db1 = db1 @ y
    db2 = db2 @ y
    return np.array([db0, db1, db2]).T


def calculate_f(y, b):
    return get_sensitivity_matrix(b) @ y


def find_params(b, t0, tk, dt, eps):
    timestamps = np.linspace(t0, tk, int((tk - t0) / dt + 1))

    while True:
        # Runge-Kutta for model
        yy = np.zeros_like(data)
        yy[0] = data[0].copy()
        for i in range(1, len(timestamps)):
            y_prev = yy[i - 1]
            k1 = dt * calculate_f(y_prev, b)
            k2 = dt * calculate_f(y_prev + k1 / 2, b)
            k3 = dt * calculate_f(y_prev + k2 / 2, b)
            k4 = dt * calculate_f(y_prev + k3, b)
            y = y_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            yy[i] = y

        # Runge-Kutta for sensitivity function
        uu = np.zeros((len(timestamps), 6, 3))
        db = calculate_model_derivatives(yy.T, b)
        A = get_sensitivity_matrix(b)
        for i in range(1, len(timestamps)):
            k1 = dt * (A @ uu[i - 1] + db[i - 1])
            k2 = dt * (A @ (uu[i - 1] + k1 / 2) + db[i - 1])
            k3 = dt * (A @ (uu[i - 1] + k2 / 2) + db[i - 1])
            k4 = dt * (A @ (uu[i - 1] + k3) + db[i - 1])
            u_next = uu[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            uu[i] = u_next

        # Finding delta b
        du = (np.array([u.T @ u for u in uu]) * dt).sum(0)
        du = np.linalg.inv(du)
        diff_y = (data - yy)
        uY = (np.array([uu[i].T @ diff_y[i] for i in range(len(timestamps))]) * dt).sum(0)
        diff_beta = du @ uY
        b += diff_beta

        if np.abs(diff_beta).max() < eps:
            break

    return b


if __name__ == "__main__":
    b0 = np.array([0.1, 0.08, 21])
    print(find_params(b0, 0, 50, 0.2, 0.001))
