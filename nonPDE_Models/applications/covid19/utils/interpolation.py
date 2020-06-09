from __future__ import absolute_import, division, print_function

import autograd.numpy as np


def piecewiseConstant(x, xp, yp):

    if np.isscalar(xp):
        if np.isscalar(x):
            y = yp
        else:
            n = len(x)
            if np.isscalar(yp):
                y = np.ones(n) * yp
            else:
                y = np.tile(yp, (n, 1))
    else:
        if np.isscalar(x):
            if x <= xp[0]:
                y = yp[0]
            elif x >= xp[-1]:
                y = yp[-1]
            else:
                i = 0
                while x >= xp[i+1]:
                    i = i + 1
                y = yp[i]
        else:
            n = len(x)

            if yp.ndim > 1:
                y = np.zeros((n, yp.shape[1]))
            else:
                y = np.zeros(n)

            i = 0
            for j in range(n):
                if x[j] <= xp[0]:
                    y[j] = yp[0]
                elif x[j] >= xp[-1]:
                    y[j] = yp[-1]
                else:
                    while x[j] >= xp[i+1]:
                        i = i + 1
                    y[j] = yp[i]

    return y


def piecewiseLinear(x, xp, yp):

    if np.isscalar(x):
        if x <= xp[0]:
            y = yp[0]
        elif x >= xp[-1]:
            y = yp[-1]
        else:
            i = 0
            while x >= xp[i+1]:
                i = i + 1
            y = (x - xp[i + 1]) / (xp[i] - xp[i + 1]) * yp[i] + (x - xp[i]) / (xp[i + 1] - xp[i]) * yp[i + 1]

    else:
        n = len(x)

        if yp.ndim > 1:
            y = np.zeros((n, yp.shape[1]))
        else:
            y = np.zeros(n)

        i = 0
        for j in range(n):
            if x[j] <= xp[0]:
                y[j] = yp[0]
            elif x[j] >= xp[-1]:
                y[j] = yp[-1]
            else:
                while x[j] >= xp[i+1]:
                    i = i + 1
                y[j] = (x[j]-xp[i+1])/(xp[i] - xp[i+1])*yp[i] + (x[j]-xp[i])/(xp[i+1] - xp[i])*yp[i+1]

    return y


def cubicSpline(x, xp, yp):

    m = len(xp)

    xdiff = np.diff(xp)
    ydiff = np.diff(yp, axis=0)

    A = np.zeros((m, m))
    A[0, 0] = 2. / xdiff[0]
    A[0, 1] = 1. / xdiff[0]
    A[m-1, m-1] = 2. / xdiff[-1]
    A[m-1, m-2] = 1. / xdiff[-1]

    # if yp.ndim > 1:
    #     b = np.zeros((m, yp.shape[1]))
    # else:
    #     b = np.zeros(m)
    #
    # b[0] = 3 * ydiff[0] / xdiff[0]**2
    # b[-1] = 3 * ydiff[-1] / xdiff[-1]**2
    #
    # for i in range(1, m-1):
    #     A[i, i-1] = 1. / xdiff[i-1]
    #     A[i, i] = 2. / xdiff[i-1] + 2. / xdiff[i]
    #     A[i, i+1] = 1. / xdiff[i]
    #     b[i] = 3 * ydiff[i-1] / xdiff[i-1]**2 + 3 * ydiff[i] / xdiff[i]**2

    # use np.append to support autograd
    b = 3 * ydiff[0] / xdiff[0]**2

    for i in range(1, m-1):
        A[i, i-1] = 1. / xdiff[i-1]
        A[i, i] = 2. / xdiff[i-1] + 2. / xdiff[i]
        A[i, i+1] = 1. / xdiff[i]
        b = np.append(b, 3 * ydiff[i-1] / xdiff[i-1]**2 + 3 * ydiff[i] / xdiff[i]**2)

    b = np.append(b, 3 * ydiff[-1] / xdiff[-1]**2)

    k = np.linalg.solve(A, b)
    # k = np.dot(np.linalg.inv(A), b)

    if np.isscalar(x):
        if x <= xp[0]:
            y = yp[0]
        elif x >= xp[-1]:
            y = yp[-1]
        else:
            i = 0
            while x >= xp[i+1]:
                i = i + 1
            t = (x - xp[i]) / xdiff[i]
            a = k[i] * xdiff[i] - ydiff[i]
            b = - k[i + 1] * xdiff[i] + ydiff[i]
            y = (1 - t) * yp[i] + t * yp[i + 1] + t * (1 - t) * ((1 - t) * a + t * b)

    else:
        n = len(x)

        if yp.ndim > 1:
            y = np.zeros((n, yp.shape[1]))
        else:
            y = np.zeros(n)

        i = 0
        for j in range(n):
            if x[j] <= xp[0]:
                y[j] = yp[0]
            elif x[j] >= xp[-1]:
                y[j] = yp[-1]
            else:
                while x[j] >= xp[i+1]:
                    i = i + 1
                t = (x[j] - xp[i]) / xdiff[i]
                a = k[i] * xdiff[i] - ydiff[i]
                b = - k[i+1] * xdiff[i] + ydiff[i]
                y[j] = (1-t) * yp[i] + t * yp[i+1] + t*(1-t)*((1-t)*a + t*b)

    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xp = np.linspace(0, 5, 6)
    # yp = np.array([xp**2, xp**3]).T
    yp = xp**2
    x = np.linspace(-1, 6, 22)
    # x = np.linspace(2, 4, 7)
    # x = -1
    print("xp = ", xp, "\n x", x)
    y_constant = piecewiseConstant(x, xp, yp)
    y_linear = piecewiseLinear(x, xp, yp)
    y_cubic = cubicSpline(x, xp, yp)

    plt.figure()
    plt.plot(xp, yp, 'rx', label="(xp, yp)")
    plt.plot(x, y_constant, 'bo', label="(x, y_constant)")
    plt.plot(x, y_linear, 'k.-', label="(x, y_linear)")
    plt.plot(x, y_cubic, 'm.-', label="(x, y_cubic)")
    plt.legend()
    plt.show()
