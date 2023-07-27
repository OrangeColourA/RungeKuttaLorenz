import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def lorenz(x, params):
    sigma = params['sigma']
    rho = params['rho']
    beta = params['beta']

    x_dt = -sigma * x[0] + sigma * x[1]
    y_dt = rho * x[0] - x[1] - x[0] * x[2]
    z_dt = x[0] * x[1] - beta * x[2]

    xdot = np.array([x_dt, y_dt, z_dt])

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


params = {'sigma': 10, 'rho': 28, 'beta': 2.66666667}
f = lambda t, x: lorenz(x, params)
x0 = np.array([0, 1, 0])

t0 = 0
tf = 35
dt = 0.01

x, t = rk4(f, x0, t0, tf, dt)

plt.figure(1)

plt.plot(t, x[1, :], color='blue', label='y')
plt.plot(t, x[2, :], color='green', label='z')
plt.plot(t, x[0, :], color='red', label='x')
plt.xlabel('Время t')
plt.grid()
plt.legend()

plt.figure(2)

plt.plot(x[0, :], x[1, :])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

plt.figure(3)

plt.plot(x[0, :], x[2, :])
plt.xlabel('x')
plt.ylabel('z')
plt.grid()

plt.figure(4)

plt.plot(x[1, :], x[2, :])
plt.xlabel('y')
plt.ylabel('z')
plt.grid()

plt.show()

# визуализация в 3д

t = np.arange(t0, tf, dt)
nt = t.size

c = np.linspace(0, 1, nt)

DATA = go.Scatter3d(x=x[0, :], y=x[1, :], z=x[2, :],
                    line=dict(color=c,
                              width=5,
                              colorscale="Electric"),
                    mode='lines')

fig = go.Figure(data=DATA)

fig.update_layout(width=1000, height=1000,
                  margin=dict(r=10, l=10, b=10, t=10),
                  paper_bgcolor='rgb(0,0,0)',
                  scene=dict(camera=dict(up=dict(x=0, y=0, z=1),
                                         eye=dict(x=0, y=1, z=1)),
                             aspectratio=dict(x=1, y=1, z=1),
                             aspectmode='manual',
                             xaxis=dict(visible=False),
                             yaxis=dict(visible=False),
                             zaxis=dict(visible=False)
                             )
                  )

fig.write_html('lorenz1.html', auto_open=True)
