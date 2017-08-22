from matplotlib import pyplot as plt
import numpy as np

Fs = 10000
Ts = 1 / Fs

t = np.arange(-10, 10, Ts)
a = 3
pulso = np.vectorize(lambda x: int(abs(x) < a))(t)
x = np.sin(t)

X = np.fft.rfft(x)
f = np.fft.rfftfreq(len(X) * 2 - 1, Ts)
w = 2 * np.pi * f

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t, x)
ax2.plot(w, X / Fs)
ax2.grid()

plt.show()
