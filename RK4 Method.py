import math
import numpy as np
import matplotlib as mpl


# прежде всего нам необходимо описать функцию правой части


def f(t, x):
    return x * t ** 2


dt = 0.1    # шаг времени
t0 = 0      # начальный момент
x0 = 1      # начальное значение
xt = x0

while t0 < 2:
    k1 = f(t0, xt)          # вычисляем k1

    t0 = t0 + dt/2          # вычисляем k2
    xt = x0 + dt/2 * k1
    k2 = f(t0, xt)

    xt = x0 + dt/2 * k2     # вычисляем k3
    k3 = f(t0, xt)

    t0 = t0 + dt/2          # вычисляем k4
    xt = x0 + dt * k3
    k4 = f(t0, xt)

    x0 = x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    print(t0, '       ', x0)
