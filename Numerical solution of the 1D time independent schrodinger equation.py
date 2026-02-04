# Numerical solution of the 1D time-independent Schrödinger equation


import numpy as np
import matplotlib.pyplot as plt


# physical constants


hbar = 1.0
m = 1.0





L = 1.0              # length of the box
N = 400              # number of spatial points
x = np.linspace(0, L, N)
dx = x[1] - x[0]





# V = 0 inside
V_box = np.zeros(N)

# Harmonic oscillator: V = (1/2) k x^2
k = 100.0
V_ho = 0.5 * k * (x - L/2)**2


# hamiltonian 


def build_hamiltonian(V):
    H = np.zeros((N, N))

    for i in range(N):
        H[i, i] = (hbar**2) / (m * dx**2) + V[i]

        if i > 0:
            H[i, i-1] = -(hbar**2) / (2 * m * dx**2)

        if i < N - 1:
            H[i, i+1] = -(hbar**2) / (2 * m * dx**2)

    return H


# to solve eigenvalue


H_box = build_hamiltonian(V_box)
H_ho = build_hamiltonian(V_ho)

# Eigenvalues and eigenvectors
E_box, psi_box = np.linalg.eigh(H_box)
E_ho, psi_ho = np.linalg.eigh(H_ho)


# waveform normalisation


def normalise(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

for i in range(3):
    psi_box[:, i] = normalise(psi_box[:, i])
    psi_ho[:, i] = normalise(psi_ho[:, i])


# Plotting


plt.figure(figsize=(8, 5))

for i in range(3):
    plt.plot(x, psi_box[:, i] + E_box[i],
             label=f"n = {i+1}")

plt.xlabel("x")
plt.ylabel("Energy + ψ(x)")
plt.title("Particle in a Box: First Three Eigenstates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

for i in range(3):
    plt.plot(x, psi_ho[:, i] + E_ho[i],
             label=f"n = {i}")

plt.xlabel("x")
plt.ylabel("Energy + ψ(x)")
plt.title("Harmonic Oscillator: First Three Eigenstates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# energies


print("Particle in a Box Energies:")
for i in range(3):
    print(f"n = {i+1}, E = {E_box[i]:.4f}")

print("\nHarmonic Oscillator Energies:")
for i in range(3):
    print(f"n = {i}, E = {E_ho[i]:.4f}")
