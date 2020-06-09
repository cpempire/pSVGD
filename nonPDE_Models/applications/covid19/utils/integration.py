from __future__ import absolute_import, division, print_function

import autograd.numpy as np


def Euler(fun, y0, t, *args, **kwargs):

    y = y0
    yi = y0
    for i in range(len(t)-1):
        dy = fun(yi, t[i], *args, **kwargs)
        yi = yi + (t[i+1]-t[i])*dy
        y = np.append(y, yi)

    y = np.reshape(y, (len(t), len(y0)))

    return y


def RK2(fun, y0, t, *args, **kwargs):

    y = y0
    yi = y0
    for i in range(len(t)-1):
        h = t[i+1]-t[i]

        y_help = yi
        t_help = t[i]
        k1 = fun(y_help, t_help, *args, **kwargs)
        y_help = yi + h * k1 / 2
        t_help = t[i] + h / 2
        k2 = fun(y_help, t_help, *args, **kwargs)
        yi = yi + h * k2

        y = np.append(y, yi)

    y = np.reshape(y, (len(t), len(y0)))

    return y


def RK4(fun, y0, t, *args, **kwargs):

    y = y0
    yi = y0
    for i in range(len(t)-1):
        h = t[i+1]-t[i]

        y_help = yi
        t_help = t[i]
        k1 = fun(y_help, t_help, *args, **kwargs)
        y_help = yi + h * k1 / 2
        t_help = t[i] + h / 2
        k2 = fun(y_help, t_help, *args, **kwargs)
        y_help = yi + h * k2 / 2
        k3 = fun(y_help, t_help, *args, **kwargs)
        y_help = yi + h * k3
        t_help = t[i] + h
        k4 = fun(y_help, t_help, *args)

        yi = yi + h/6*(k1+2.*k2+2.*k3+k4)

        y = np.append(y, yi)

    y = np.reshape(y, (len(t), len(y0)))

    return y