import numpy as np
import matplotlib.pyplot as plt


# определение системы дифференциальных уравнений


def LVm(x, params):

    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']

    xdot = np.array([alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])

    return xdot


# описание алгоритма Рунге-Кутта


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


# решение

params = {'alpha': 1.1, 'beta': 0.4, 'gamma': 0.3, 'delta': 0.1}

f = lambda t, x: LVm(x, params)

x0 = np.array([20, 5])

t0 = 0
tf = 100
dt = 0.01

x, t = rk4(f, x0, t0, tf, dt)

# plot results

plt.subplot(1, 2, 1)
plt.plot(t, x[0, :], color='red', label='preys')
plt.plot(t, x[1, :], color='blue', label='predators')
plt.xlabel('Время t')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x[0, :], x[1, :])
plt.xlabel('Preys')
plt.ylabel('Predators')
plt.grid()

plt.savefig('test.png')
plt.show()
