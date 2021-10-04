import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# Reading data from file
data_str = open("./f6.txt").read().split()
data = np.array(data_str, float)

# Preparing array with timestamps
T = 5
dt = 0.01
time = np.arange(0, T + dt, dt)

# Finding Fourier transforms
n = time.shape[0]
N = np.arange(n)
k = N.reshape((n, 1))
M = np.exp(-2j * np.pi * k * N / n)

transformed_data = np.dot(M, data) / n
transformed_half = transformed_data[:transformed_data.shape[0] // 2 - 1]
extremums = np.array(argrelextrema(transformed_half, np.greater))
main_frequency = extremums[0][0]/T
print(main_frequency)

# Apply least squares method to find resulted approximation
b = np.array([np.sum(data * time ** 3), np.sum(data * time ** 2), np.sum(data * time),
              np.sum(data * np.sin(2. * np.pi * main_frequency * time)), np.sum(data)])

a = np.zeros((b.shape[0], b.shape[0]))

functions = [time ** 3, time ** 2, time, np.sin(2. * np.pi * main_frequency * time), np.ones(n)]

for i in range(b.shape[0]):
    for j in range(b.shape[0]):
        a[i, j] = np.sum(functions[i] * functions[j])

solution = np.matmul(np.linalg.inv(a), b.T)     # <- resulted (a1, a2,..., an)  coefficients

print(solution)
approximated_func = np.dot(solution, functions)     # <- approximated function by its values

# Displaying data from all steps. Also we could see that function approximation was done well
fig, axs = plt.subplots(2)
plt.grid(True)
fig.suptitle("Data, Frequencies and Resulted approximation")
axs[0].plot(time, data)
axs[1].plot(time, approximated_func)
plt.show()
