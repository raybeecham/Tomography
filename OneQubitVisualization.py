import numpy as np
import matplotlib.pyplot as plt

# plot the density matrix
def plot_density_matrix(rho, title="Density Matrix"):
    fig, ax = plt.subplots()
    cax = ax.matshow(np.abs(rho), cmap='viridis')
    for (i, j), val in np.ndenumerate(rho):
        ax.text(j, i, f"{np.abs(val):0.2f}\n{np.angle(val):0.2f}π", ha='center', va='center', color='white')
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    plt.show()

# Hadamard gate
def hadamard():
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

# Apply Z rotation gate
def apply_z_rotation(rho, phi):
    # Z rotation matrix
    Rz = np.array([[np.exp(-1j * phi / 2), 0],
                   [0, np.exp(1j * phi / 2)]])
    return Rz @ rho @ Rz.conj().T

# Initial state |0><0| (pure state)
p = 1.0
rho = np.array([
    [p, 0],
    [0, 1-p]
])

# Plot the initial state
plot_density_matrix(rho, "Original Density Matrix")

# Apply first Hadamard
H = hadamard()
rho_after_h = H @ rho @ H.conj().T
plot_density_matrix(rho_after_h, "After First Hadamard")

# Apply Z rotation
phi = np.pi  # 180 degrees, which flips the sign of |1>
rho_after_z = apply_z_rotation(rho_after_h, phi)
plot_density_matrix(rho_after_z, "After Z Rotation")

# Apply second Hadamard
rho_after_second_h = H @ rho_after_z @ H.conj().T
plot_density_matrix(rho_after_second_h, "After Second Hadamard")
