import numpy as np

# Equation = 2sin(x/2) + 5
x = np.linspace(0, 2*np.pi, 1000)
y = 2*np.sin(x/2) + 5
noise = np.random.normal(0, 1, 1000)

np.save(r"\\wsl.localhost\Ubuntu\home\tanji\Denario-SR\datasets\SHM_1\time.npy", x)
np.save(r"\\wsl.localhost\Ubuntu\home\tanji\Denario-SR\datasets\SHM_1\displacement.npy", y+noise)

