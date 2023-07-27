import numpy as np
import matplotlib.pyplot as plt


def test(x):
    xdot = np.array([-x[1], x[0]])

    return xdot


def rk4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size

    nx = x0.size
    x = np.zeros((nx, nt))

    x[:, 0] = x0

    for k in range(nt - 1):
        k1 = f(t[k], x[:, k])
        k2 = f(t[k] + dt / 2, x[:, k] + dt * k1 / 2)
        k3 = f(t[k] + dt / 2, x[:, k] + dt * k2 / 2)
        k4 = f(t[k] + dt, x[:, k] + dt * k3)

        dx = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x[:, k + 1] = x[:, k] + dx

    return x, t


f = lambda t, x: test(x)

x0 = np.array([0, 20])
t0 = 0
tf = 20
dt = 0.01
x, t = rk4(f, x0, t0, tf, dt)

plt.subplot(1, 2, 1)
plt.plot(t, x[0, :], color='red')
plt.plot(t, x[1, :], color='blue')
plt.xlabel('Время t')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x[0, :], x[1, :])
plt.grid()

plt.savefig('test1.png')
plt.show()
