import matplotlib.pyplot as plt
import numpy as np

devices_nonunified = ["CPU", "GPU <1, 1>", "GPU <1, 256>", "GPU <N, 256>"]
devices_unified = ["CPU", "GPU <1, 1>", "GPU <1, 256>", "GPU <N, 256>"]

x_axis_nonunified = np.arange(len(devices_nonunified))
x_axis_unified = np.arange(len(devices_unified))

# get the non-unified times (ms)
time_nonunified_1 = [5, 92.203, 1.5171, 1.3742]
time_nonunified_5 = [20, 391.75, 8.0127, 8.0145]
time_nonunified_10 = [40, 721.50, 16.314, 16.321]
time_nonunified_50 = [220, 3609.69, 82.121, 82.779]
time_nonunified_100 = [450, 7216.3, 165.90, 165.73]

# get the unified times (ms)
time_unified_1 = [5, 85.737, 2.5556, 2.0643]
time_unified_5 = [20, 355.48, 12.427, 9.6329]
time_unified_10 = [40, 645.66, 24.317, 17.963]
time_unified_50 = [220, 3226.9, 91.38, 122.69]
time_unified_100 = [450, 6447.7, 169.36, 228.70]

# bars for x axis - non unified
plt.bar(x_axis_nonunified + 0.1, time_nonunified_1, 0.1, label="1")
plt.bar(x_axis_nonunified + 0.2, time_nonunified_5, 0.1, label="5")
plt.bar(x_axis_nonunified + 0.3, time_nonunified_10, 0.1, label="10")
plt.bar(x_axis_nonunified + 0.4, time_nonunified_50, 0.1, label="50")
plt.bar(x_axis_nonunified + 0.5, time_nonunified_100, 0.1, label="100")

plt.xticks(x_axis_nonunified, devices_nonunified)
plt.xlabel("Devices and Threads")
plt.ylabel("Time (ms) - log scale")
plt.yscale("log")
plt.title("Time required to add two arrays on cpu/device memory")
plt.legend()
plt.show()

# bars for x axis - unified
plt.bar(x_axis_unified + 0.1, time_unified_1, 0.1, label="1")
plt.bar(x_axis_unified + 0.2, time_unified_5, 0.1, label="5")
plt.bar(x_axis_unified + 0.3, time_unified_10, 0.1, label="10")
plt.bar(x_axis_unified + 0.4, time_unified_50, 0.1, label="50")
plt.bar(x_axis_unified + 0.5, time_unified_100, 0.1, label="100")

plt.xticks(x_axis_unified, devices_unified)
plt.xlabel("Devices and Threads")
plt.ylabel("Time (ms) - log scale")
plt.yscale("log")
plt.title("Time required to add two arrays on unified memory")
plt.legend()
plt.show()
