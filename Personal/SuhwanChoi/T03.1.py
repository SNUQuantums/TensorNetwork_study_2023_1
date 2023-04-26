import numpy as np

# We will decompose tensor with QR, SVD and calculate entanglement entropy of decomposed tensors.

# We will use the following notation:
#   1. A = QR, where Q is unitary and R is upper triangular.
#   2. A = U S V, where U and V are unitary and S is diagonal.

N = 5
sigma = [*range(2, N + 2)]
T0 = T = np.random.rand(*sigma)
Q = []
alpha = 1

for i in range(N - 1):
    T = T.reshape((alpha * sigma[i], -1))
    U, S, VT = np.linalg.svd(T, full_matrices=False)
    print(U.shape, S.shape, VT.shape)
    U = U.reshape((alpha, sigma[i], -1))
    Q.append(U)
    T = S.reshape((-1, 1)) * VT
    alpha = U.shape[-1]
T = T.reshape((alpha, sigma[N - 1], -1))
Q.append(T)

print("[ Shapes ]")
for q in Q:
    print(q.shape)

print(T0[1,1,1,1,1])

q0 = Q[0][:, 1, :]
q1 = Q[1][:, 1, :]
q2 = Q[2][:, 1, :]
q3 = Q[3][:, 1, :]
q4 = Q[4][:, 1, :]
q = q0 @ q1 @ q2 @ q3 @ q4

print(q)


q = np.tensordot(Q[0], Q[1], axes=([-1], [0]))
q = np.tensordot(q, Q[2], axes=([-1], [0]))
q = np.tensordot(q, Q[3], axes=([-1], [0]))
q = np.tensordot(q, Q[4], axes=([-1], [0]))
q = q.squeeze((0, -1))
print(q[1, 1, 1, 1, 1])
print(np.sum(np.abs(q - T0)))