import torch

# We will decompose tensor with QR, SVD and calculate entanglement entropy of decomposed tensors.

# We will use the following notation:
#   1. A = QR, where Q is unitary and R is upper triangular.
#   2. A = U S V, where U and V are unitary and S is diagonal.

print("# Exercise (a): Check the integrity of the tensor decomposition")

N = 5
sigma = [*range(6, N + 6)]
# T0 = T = np.random.rand(*sigma)
T0 = T = torch.rand(*sigma, dtype=torch.complex128) # float64 + float64 > complex128.
Qs = []
alpha = 1

for i in range(N - 1):
    T = T.reshape((alpha * sigma[i], -1))
    # U, S, VT = np.linalg.svd(T, full_matrices=False)
    # U, S, VT = torch.linalg.svd(T, full_matrices=False)
    Q, R = torch.linalg.qr(T, mode='reduced')
    Q = Q.reshape((alpha, sigma[i], -1))
    Qs.append(Q)
    T = R
    alpha = Q.shape[-1]
T = T.reshape((alpha, sigma[N - 1], -1))
Qs.append(T)

print("Shapes:", [tuple(Q.shape) for Q in Qs])
print("T0[1,1,1,1,1]:", T0[1,1,1,1,1])

# First way to get `T[1,1,1,1,1]`
q0 = Qs[0][:, 1, :]
q1 = Qs[1][:, 1, :]
q2 = Qs[2][:, 1, :]
q3 = Qs[3][:, 1, :]
q4 = Qs[4][:, 1, :]
q = q0 @ q1 @ q2 @ q3 @ q4

print("T0[1,1,1,1,1]:", q[0, ..., 0])

# Second way to get `T[1,1,1,1,1]` (you can also use `torch.einsum`)
q = torch.tensordot(Qs[0], Qs[1], ([-1], [0]))
q = torch.tensordot(q, Qs[2], ([-1], [0]))
q = torch.tensordot(q, Qs[3], ([-1], [0]))
q = torch.tensordot(q, Qs[4], ([-1], [0]))
q = q.squeeze(0).squeeze(-1)
# q = q.squeeze((0, -1)) # for numpy
print("T0[1,1,1,1,1]:", q[1, 1, 1, 1, 1])
# double precision noise ~ 53 bits ~ 16 digits = 1e-16
print("difference max:", torch.max(torch.abs(q - T0)))

print("# Exercise (b): Entanglement entropies for different bipartitions")

for A in [{0, 1}, {0, 2}, {0, 4}]:
    B = list(set(range(N)) - A)
    A = list(A)
    tmp = T0.permute(*A, *B).contiguous().view(sigma[A[0]] * sigma[A[1]], -1)
    U, S, VT = torch.linalg.svd(tmp, full_matrices=False)
    S /= torch.sqrt(torch.sum(S ** 2))
    print(f"Entanglement entropy for bipartition {A}, {B} is {-torch.sum(S ** 2 * torch.log2(S ** 2)):.4f}")


print("# Exercise (c): Use the SVD for the tensor decomposition and compute the entanglement entropy")

T = T0
Us = []
alpha = 1

for i in range(N - 1):
    T = T.reshape((alpha * sigma[i], -1))
    U, S, VT = torch.linalg.svd(T, full_matrices=False)
    s = S / torch.sqrt(torch.sum(S ** 2))
    print(f"Entanglement entropy for bipartition for {i}th SVD is: {-torch.sum(s ** 2 * torch.log2(s ** 2)):.4f}")
    U = U.reshape((alpha, sigma[i], -1))
    Us.append(U)
    T = S.reshape((-1, 1)) * VT
    alpha = U.shape[-1]
T = T.reshape((alpha, sigma[N - 1], -1))
Us.append(T)

# Yes, this is `torch.einsum`
q = torch.einsum('abc, cde, efg, ghi, ijk -> abdfhjk', Us[0], Us[1], Us[2], Us[3], Us[4])
q = q.squeeze(0).squeeze(-1)
print("Params:", sum([torch.numel(U) for U in Us]))
print("Shapes:", [tuple(U.shape) for U in Us])
print("T0[1,1,1,1,1]:", q[1, 1, 1, 1, 1])
print("difference max:", torch.max(torch.abs(q - T0)))

T = T0
Us = []
alpha = 1
eps = 5

for i in range(N - 1):
    T = T.reshape((alpha * sigma[i], -1))
    U, S, VT = torch.linalg.svd(T, full_matrices=False)
    # Caution: SVD of matrix is not unique.
    # especially, spectral values are given **nonnegative** in pytorch and numpy.
    # SVD is given by eigendecomposition of A^T A, which is positive semidefinite, which gives nonnegative eigenvalues.
    # singular value is square root of eigenvalues of A^T A
    # Spectral values may not be nonnegative in other variants of SVD(randomized SVD, ...)
    idx = S < eps
    U = U[:, ~idx]
    S = S[~idx]
    VT = VT[~idx, :]
    U = U.reshape((alpha, sigma[i], -1))
    Us.append(U)
    T = S.reshape((-1, 1)) * VT
    alpha = U.shape[-1]
T = T.reshape((alpha, sigma[N - 1], -1))
Us.append(T)

q = torch.einsum('abc, cde, efg, ghi, ijk -> abdfhjk', Us[0], Us[1], Us[2], Us[3], Us[4])
q = q.squeeze(0).squeeze(-1)
print("Params:", sum([torch.numel(U) for U in Us]))
print("Shapes:", [tuple(U.shape) for U in Us])
print("T0[1,1,1,1,1]:", q[1, 1, 1, 1, 1])
print("difference max:", torch.max(torch.abs(q - T0)))