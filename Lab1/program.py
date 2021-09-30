import matplotlib.pyplot as plt
import numpy as np

# Reading data from file
data_str = open("./f6.txt").read().split()
data = np.array(data_str, float)

# Preparing array with timestamps
T = 5
dt = 0.01
time = np.arange(0, T + dt, dt)

# Finding Fourier transforms
n = time.shape[0]
frequency = np.zeros(n)

for point_id in range(n):
    sin_freq = 0
    cos_freq = 0

    for signal in range(n):
        sin_freq += data[signal] * np.sin(2. * np.pi * point_id * signal / float(n))
        cos_freq += data[signal] * np.cos(2. * np.pi * point_id * signal / float(n))

    sin_freq /= float(n)
    cos_freq /= float(n)

    frequency[point_id] = np.sqrt(sin_freq ** 2 + cos_freq ** 2)

# Finding local maximum
biggest_value = []

for i in range(3, n // 2):
    if np.max(frequency[i - 3:i + 3]) == frequency[i]:
        biggest_value.append(i)
        print(frequency[i])

main_frequency = biggest_value[0] / T   # <- local maximum

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
fig, axs = plt.subplots(3)
plt.grid(True)
fig.suptitle("Data, Frequencies and Resulted approximation")
axs[0].plot(time, data)
axs[1].plot(time, frequency)
axs[2].plot(time, approximated_func)
plt.show()
