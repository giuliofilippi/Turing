#∂A/∂t = D_A∇^2A - A*B^2 + f(1 - A)
#∂B/∂t = D_B∇^2B + A*B^2 - (k + f) * B

# imports
import numpy as np
import matplotlib.pyplot as plt

# derivatives
def dAdt(A, B, Da, Db, f, k):
    laplace_A = np.roll(A, shift=(1, 0), axis=(0, 1)) + np.roll(A, shift=(-1, 0), axis=(0, 1)) + \
                np.roll(A, shift=(0, 1), axis=(0, 1)) + np.roll(A, shift=(0, -1), axis=(0, 1)) - 4 * A
    return Da * laplace_A - A * B ** 2 + f * (1 - A)

def dBdt(A, B, Da, Db, f, k):
    laplace_B = np.roll(B, shift=(1, 0), axis=(0, 1)) + np.roll(B, shift=(-1, 0), axis=(0, 1)) + \
                np.roll(B, shift=(0, 1), axis=(0, 1)) + np.roll(B, shift=(0, -1), axis=(0, 1)) - 4 * B
    return Db * laplace_B + A * B ** 2 - (k + f) * B

# Set the size of your grid
grid_size = (100, 100)

# Initialize A and B grids
A = np.ones(grid_size)
B = np.zeros(grid_size)

# You can set some initial conditions, e.g., adding a seed in the middle:
A[45:55, 45:55] = 0.5   #0.5
B[45:55, 45:55] = 0.25  #0.25

# Set your simulation parameters
Da = 0.2
Db = 0.1
f = 0.04
k = 0.06
dt = 1.0
num_steps = 2000   # 10000

for _ in range(num_steps):
    dA = dAdt(A, B, Da, Db, f, k)
    dB = dBdt(A, B, Da, Db, f, k)

    A += dA * dt
    B += dB * dt

plt.imshow(B, cmap='Blues')
plt.colorbar()
plt.show()