import numpy as np
import plotly.graph_objects as go


def wang_sun(x, params):
    alpha = params['alpha']
    beta = params['beta']
    zeta = params['zeta']
    delta = params['delta']
    epsilon = params['epsilon']
    xi = params['xi']

    x_dt = alpha * x[0] + zeta * x[1] * x[2]
    y_dt = beta * x[0] + delta * x[1] - x[0] * x[2]
    z_dt = epsilon * x[2] + xi * x[0] * x[1]

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


params = {'alpha': 0.2, 'beta': -0.01, 'zeta': 1, 'delta': -0.4, 'epsilon': -1, 'xi': -1}

f = lambda t, x: wang_sun(x, params)

x0 = np.array([0.5, 0.1, 0.1])

t0 = 0
tf = 500
dt = 0.01

x, t = rk4(f, x0, t0, tf, dt)

t = np.arange(t0, tf, dt)
nt = t.size

c = np.linspace(0, 1, nt)

DATA = go.Scatter3d(x=x[0, :], y=x[1, :], z=x[2, :],
                    line=dict(color=c,
                              width=5,
                              colorscale="Hot"),
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
