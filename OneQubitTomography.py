# Description: This script demonstrates how to perform quantum tomography on a single qubit state using Python.
# The QuantumTomography class contains methods to construct the density matrix from counts, apply the HZH sequence, calculate fidelity, and calculate concurrence.
# The example constructs a QuantumTomography object with counts corresponding to |0> and |1>, applies the HZH sequence, and calculates the fidelity and concurrence of the original state.
# The output shows the original and transformed density matrices, the fidelity between the original and transformed states, and the concurrence of the original state.
# The QuantumTomography class can be extended to handle more complex quantum states and operations.

import numpy as np
from scipy.linalg import fractional_matrix_power

# create a class to perform quantum tomography on a single qubit
class QuantumTomography:
    def __init__(self, counts, basis):
        self.counts = counts
        self.basis = basis
        self.rho = self.construct_rho()

    # construct the density matrix from the counts
    def construct_rho(self):
        # Simplified density matrix construction for a single qubit
        norm = np.sum(self.counts)
        rho = np.array([
            [self.counts[0] / norm, 0],  # Probability amplitude for |0>
            [0, self.counts[1] / norm]   # Probability amplitude for |1>
        ])
        return rho

    # Hadamard gate
    def hadamard(self):
        return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    # Z rotation gate
    def apply_z_rotation(self, phi):
        # Z rotation matrix
        Rz = np.array([[np.exp(-1j * phi / 2), 0],
                       [0, np.exp(1j * phi / 2)]])
        return Rz @ self.rho @ Rz.conj().T

    # Apply the HZH sequence to the density matrix
    def apply_hzh_sequence(self):
        # Apply Hadamard, Z rotation (phi = pi), and another Hadamard
        H = self.hadamard()
        intermediate_rho = H @ self.rho @ H.conj().T  # First Hadamard
        intermediate_rho = self.apply_z_rotation(np.pi) @ intermediate_rho @ self.apply_z_rotation(np.pi).conj().T  # Z rotation
        final_rho = H @ intermediate_rho @ H.conj().T  # Second Hadamard
        return final_rho

    # Calculate the fidelity between the current state and another density matrix
    def calculate_fidelity(self, other_rho):
        sqrt_rho = fractional_matrix_power(self.rho, 0.5)
        product = sqrt_rho @ other_rho @ sqrt_rho
        return np.real(np.trace(fractional_matrix_power(product, 0.5))**2)

    # Calculate the concurrence of the current state
    def calculate_concurrence(self):
        return 0  # Single qubit states do not have concurrence

# Example
counts = np.array([10, 5])  # counts corresponding to |0> and |1>
basis = ["0", "1"]

# Create a QuantumTomography object
qt = QuantumTomography(counts, basis)
print("Original Density Matrix:\n", qt.rho)

# Apply the HZH sequence
transformed_rho = qt.apply_hzh_sequence()
print("Transformed Density Matrix:\n", transformed_rho)

# Calculate fidelity and concurrence
fidelity = qt.calculate_fidelity(transformed_rho)
print("Fidelity between original and transformed states:", fidelity)

concurrence = qt.calculate_concurrence()
print("Concurrence of the original state:", concurrence)

# Output: 
"""
Original Density Matrix:
 [[0.66666667 0.        ]
 [0.         0.33333333]]
Transformed Density Matrix:
 [[0.17592593+0.j 0.08333333+0.j]
 [0.08333333+0.j 0.10185185+0.j]]
Fidelity between original and transformed states: 0.2500000000000011
Concurrence of the original state: 0
"""
# The output shows the original and transformed density matrices, the fidelity between the original and transformed states, and the concurrence of the original state.
# The original density matrix corresponds to a state with probabilities 2/3 for |0> and 1/3 for |1>.
# The transformed density matrix shows the values 0.1759, 0.0833, 0.0833, and 0.1019, which are the probabilities of the transformed state.
# The fidelity between the original and transformed states is approximately 0.25, indicating the similarity between the two states.
# Since the original state is a single qubit state, the concurrence is 0, as single qubit states do not exhibit entanglement.
