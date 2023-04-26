import numpy as np, time

d_a = 101; # d_alpha
d_b = 102; # d_beta
d_c = 103; # d_gamma
d_d = 104; # d_delta
d_m = 105; # d_mu

A = np.random.rand(d_c,d_d); # tensor A(gamma,delta)
B = np.random.rand(d_a,d_m,d_c); # tensor B(alpha,mu,gamma)
C = np.random.rand(d_b,d_m,d_d); # tensor C(beta,mu,delta)

now = time.time()

B1 = B.transpose((0, 2, 1)).reshape((d_a*d_c,d_m))
C1 = C.transpose((1, 0, 2)).reshape((d_m,d_b*d_d))
BC = B1 @ C1
BC = BC.reshape((d_a,d_c,d_b,d_d))
BC = BC.transpose((0, 2, 1, 3)).reshape((d_a*d_b,d_c*d_d))
A1 = A.flatten()

ABC1 = BC @ A1
ABC1 = ABC1.reshape((d_a,d_b))

print(time.time() - now) # 0.5828

now = time.time()
np.einsum('cd,amc,bmd->ab', A, B, C, optimize=True)
print(time.time() - now) # 0.0247

now = time.time()
np.einsum('cd,amc,bmd->ab', A, B, C, optimize=False)
print(time.time() - now) # 20.24

# (AC)B:
#   expression. (left dim)(right dim) = O(complexity), where parenthesis is on contracted legs.
#   1. (cd)(bmd) = bc(d)m
#   2. (bcm)(amc) = ab(cm)
#   sum = bcdm + abcm
# (BC)A:
#   1. (amc)(bmd) = abc(m)d
#   2. (abcd)(cd) = ab(cd)
#   sum = abcdm + abcd
# (AC)B is better then (BC)A.

now = time.time()
AC = np.einsum("cd,bmd->bcm", A, C)
np.einsum("bcm,amc->ab", AC, B)
print(time.time() - now) # 0.2481 (on intel 10700)

now = time.time()
BC = np.einsum("amc,bmd->abcd", B, C)
np.einsum("abcd,cd->ab", BC, A)
print(time.time() - now) # 23.18 (on intel 10700)

# About 100 times faster, which is expected!

now = time.time()
AC = np.einsum("cd,bmd->bcm", A, C, optimize=True)
np.einsum("bcm,amc->ab", AC, B, optimize=True)
print(time.time() - now) # 0.0200 (on intel 10700)

now = time.time()
BC = np.einsum("amc,bmd->abcd", B, C, optimize=True)
np.einsum("abcd,cd->ab", BC, A, optimize=True)
print(time.time() - now) # 1.477 (on intel 10700)

# About 100 times faster, which is expected, but not just right as previous case;
# it's because optimization made some nonlinearity.
