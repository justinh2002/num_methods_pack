import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

# Parameters
Lx = 2 * np.pi  # Length in x-direction
Ly = 1.0        # Length in y-direction
Nx = 16         # Number of Fourier modes in x
Ny = 16         # Number of Chebyshev modes in y
alpha = 0.01    # Diffusivity constant
T = 10.0         # Total time
dt = 0.01       # Time step

# Grids for x and y
x = np.linspace(0, Lx, Nx, endpoint=False)  # Fourier grid in x
y = np.cos(np.pi * np.arange(Ny) / (Ny - 1)) * Ly / 2  # Chebyshev grid in y

# Fourier wave numbers for x
kx = np.fft.fftfreq(Nx, d=Lx / (2 * np.pi * Nx))

# Differentiation matrix in Chebyshev space (2nd derivative)
D = np.zeros((Ny, Ny))
for i in range(1, Ny - 1):
    D[i, i] = -2.0
    D[i, i - 1] = D[i, i + 1] = 1.0
D[0, 0] = D[-1, -1] = 1.0

# Scale differentiation matrix for y-direction Laplacian
Dy2 = (4 / Ly**2) * D

# Initial condition (e.g., a Gaussian blob)
X, Y = np.meshgrid(x, y)
u0 = np.exp(-10 * ((X - Lx / 2)**2 + (Y - 0.5 * Ly)**2))

# Fourier transform in x-direction
u_hat = np.fft.fft(u0, axis=1)

# Time evolution
u_hat_time = u_hat.copy()
t = 0
while t < T:
    for j in range(Nx):  # Loop over Fourier modes in x
        u_hat_time[:, j] = np.linalg.solve(np.eye(Ny) + alpha * dt * (kx[j]**2) * np.eye(Ny) - alpha * dt * Dy2, u_hat[:, j])
    t += dt
    u_hat = u_hat_time

# Transform back to physical space
u_final = np.fft.ifft(u_hat, axis=1).real

# Plot initial and final solution
plt.subplot(1, 2, 1)
plt.contourf(X, Y, u0, 20, cmap='hot')
plt.title('Initial condition')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, u_final, 20, cmap='hot')
plt.title(f'Solution at t = {T}')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
