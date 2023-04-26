import numpy as np

def hamiltonian(t):
    H = np.diag(-t, -1)
    H += np.matrix.getH(H)
    return H
def ground_state(H):
    w, v = np.linalg.eigh(H)
    w[abs(w) < 1e-9] = 0
    return np.sum(w[w < 0]), 2 ** np.sum(w == 0), w

for t in [np.ones(9), np.ones(10), np.exp(1j*np.arange(1,10+1))]:
    H = hamiltonian(t)
    print(ground_state(H)[:2])