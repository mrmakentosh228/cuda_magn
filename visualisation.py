import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


cudapath = "cuda_app\mf.txt"
cpupath = "cpu_app\mf.txt"
f = open(cudapath, "r")
f1 = open(cpupath, "r")
ar = f.readlines()
arr = np.array([np.array(list(map(float, x.split()))) for x in ar])
ar1 = f.readlines()
arr1 = np.array([np.array(list(map(float, x.split()))) for x in ar])
f.close()
f1.close()

n = 256
h = 0.004
x = np.linspace(0, h * n, n)
y = np.linspace(0, h * n, n)
xg, yg = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(xg, yg, arr, cmap='inferno')
ax.set_title("GPU")

ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.plot_surface(xg, yg, arr1, cmap='inferno')
ax1.set_title("CPU")
plt.show()

