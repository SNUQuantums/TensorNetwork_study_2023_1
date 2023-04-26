import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("Gwanghwamun.jpg").convert("L")
U, S, VT = np.linalg.svd(img)
S = np.diag(S)
for i, j in enumerate([1, 3, 10, 30, 100, 300]):
    plt.subplot(2, 3, i + 1)
    plt.imshow(np.dot(U[:, :j], np.dot(S[:j, :j], VT[:j, :])), cmap='gray')
    plt.axis('off')
plt.savefig("result.png")
plt.close()

Uf = np.fft.fft(U, axis=0)
Vf = np.fft.fft(VT.T, axis=0)
S = np.diag(S)
freq = np.fft.fftfreq(Uf.shape[0])
Uf = Uf[freq > 0, :]
freq = freq[freq > 0]
for i in [0, 1, 2]:
    plt.plot(freq, np.abs(Uf[:, i]), label=f"Uf[{i}], S={S[i]}", linewidth=0.5)

for i in [0, 1, 2]:
    i = S.shape[0] - 1 - i
    plt.plot(freq, np.abs(Uf[:, i]), label=f"Uf[{i}], S={S[i]}", linewidth=0.5)
plt.legend()
plt.savefig("fftU.png")
plt.close()

freq = np.fft.fftfreq(Vf.shape[0])
Vf = Vf[freq > 0, :]
freq = freq[freq > 0]
for i in [0, 1, 2]:
    plt.plot(freq, np.abs(Vf[:, i]), label=f"Vf[{i}], S={S[i]}", linewidth=0.5)

for i in [0, 1, 2]:
    i = S.shape[0] - 1 - i
    plt.plot(freq, np.abs(Vf[:, i]), label=f"Vf[{i}], S={S[i]}", linewidth=0.5)
plt.legend()
plt.savefig("fftV.png")
plt.close()