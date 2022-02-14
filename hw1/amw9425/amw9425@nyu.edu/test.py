import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0, 10, 0.001)
b = np.sin(2 * np.pi * a)

plt.plot(a, b)