
from __future__ import absolute_import, division, print_function

import numpy as np


class _FiniteDimensionalR:

    def __init__(self, CovInv, dimension, mpi_comm=None, diag=False):

        self.CovInv = CovInv
        self.dimension = dimension
        self.mpi_comm = mpi_comm
        self.diag = diag
        if diag:
            self.diag_ConInv = np.diag(self.CovInv)

    def init_vector(self, x, dimension):
        x.init(self.dimension)

    def mpi_comm(self):
        return self.mpi_comm

    def inner(self, x, y):

        return np.dot(y, self.CovInv.dot(x))

    def mult(self, x, y):
        if self.diag:
            y[:] = np.multiply(self.diag_ConInv, x)
        else:
            y[:] = self.CovInv.dot(x)


class _FiniteDimensionalRsolver:

    def __init__(self, Cov, dimension):

        self.Cov = Cov
        self.dimension = dimension

    def init_vector(self, x, dimension):
        x.init(self.dimension)

    def solve(self, x, b):
        x[:] = self.Cov.dot(b)

    def mult(self, x, y):
        y[:] = self.Cov.dot(x)


class _FiniteDimensionalM:

    def __init__(self, dimension):

        self.dimension = dimension

    def init_vector(self, x, dimension):
        x.init(self.dimension)

    def mult(self, x, y):
        y[:] = x


class _FiniteDimensionalMsolver:

    def __init__(self, dimension):

        self.dimension = dimension

    def init_vector(self, x, dimension):
        x.init(self.dimension)

    def solve(self, x, b):
        x[:] = b


class FiniteDimensionalPrior:

    def __init__(self, dimension, sigma=None, mean=None, uniform=False):

        self.dimension = dimension
        self.uniform = uniform

        # a vector of standard deviations
        if sigma is None:
            self.sigma = np.ones(dimension)
        else:
            self.sigma = sigma
        self.diag_CovInv = 1./(self.sigma**2)

        # mean of the prior
        if mean is None:
            self.mean = np.zeros(dimension)
        else:
            self.mean = mean

        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # self.nproc = comm.Get_size()
        # self.dimension *= self.nproc
        self.Cov = np.diag(self.sigma**2)
        self.CovSqrt = np.diag(self.sigma)
        self.CovInv = np.diag(1./(self.sigma**2))

        self.M = _FiniteDimensionalM(self.dimension)
        self.Msolver = _FiniteDimensionalMsolver(self.dimension)
        self.R = _FiniteDimensionalR(self.CovInv, self.dimension, diag=True)
        self.Rsolver = _FiniteDimensionalRsolver(self.Cov, self.dimension)

        self.Rd = np.zeros_like(self.mean)

    def init_vector(self, x, dimension):

        x.init(self.dimension)
        # print("x", len(x.get_local()), "self.dimension", self.dimension)

    def sample(self, noise, s, add_mean=True):
        s[:] = 0.
        # print("size noise = ", len(noise.get_local()), "size s = ", len(s.get_local()))
        s += self.CovSqrt.dot(noise)

        if add_mean:
            s += self.mean

    def pointwise_variance(self, method, k=1000000, r=200):

        pw_var = np.diag(self.Cov)

        return pw_var

    def trace(self):

        return np.sum(np.diag(self.Cov))

    def cost(self, m):

        if self.uniform:
            return 0.
        else:
            d = m - self.mean
            self.R.mult(d, self.Rd)
            return .5*np.dot(self.Rd, d)

    def grad(self, m, out):
        if self.uniform:
            out[:] = 0.
        else:
            d = m - self.mean
            self.R.mult(d, out)


class BrownianMotion(FiniteDimensionalPrior):

    def __init__(self, dimension):

        self.dimension = dimension

        self.t = np.linspace(0, 1, dimension+1)

        self.mean = np.zeros(dimension)

        self.Cov = np.zeros((dimension, dimension))

        for i in range(dimension):
            for j in range(dimension):
                self.Cov[i,j] = np.min([self.t[i+1], self.t[j+1]])

        d, U = np.linalg.eigh(self.Cov)
        self.d_Cov = d
        self.CovSqrt = np.dot(U, np.diag(np.sqrt(d))).dot(U.T)
        dinv = d.copy()
        dinv[1:] = 1./d[1:]
        self.CovInv = np.dot(U, np.diag(dinv)).dot(U.T)

        self.M = _FiniteDimensionalM(self.dimension)
        self.Msolver = _FiniteDimensionalMsolver(self.dimension)
        self.R = _FiniteDimensionalR(self.CovInv, self.dimension)
        self.Rsolver = _FiniteDimensionalRsolver(self.Cov, self.dimension)

    # def sample(self, noise, s, add_mean=True):
    #     s[:] = 0.
    #     # print("size noise = ", len(noise.get_local()), "size s = ", len(s.get_local()))
    #     s += self.CovSqrt.dot(noise)
    #
    #     if add_mean:
    #         s += self.mean


class Laplacian(FiniteDimensionalPrior):

    def __init__(self, dimension, delta=0., gamma=10., h=1, mean=None, regularization=True):
        """
        Gaussian prior with covariance as C = (-gamma * Laplace + delta * I)^{-1}
        :param dimension:
        :param delta:
        :param gamma:
        :param h: step size
        :param mean: mean
        """
        # mean of the prior
        if mean is None:
            self.mean = np.zeros(dimension)
        else:
            self.mean = mean

        self.regularization = regularization

        self.dimension = dimension
        Laplace = np.eye(dimension, dimension, k=-1) - 2 * np.eye(dimension, dimension, k=0) + np.eye(dimension, dimension, k=1)
        Laplace = 1./h**2 * Laplace
        # d, U = np.linalg.eigh(Laplace)
        # print("Laplace = ", Laplace, "d, U = ", d, U)

        Mass = np.eye(dimension, dimension, k=0)
        self.CovInv = - gamma * Laplace + delta * Mass
        d, U = np.linalg.eigh(self.CovInv)
        dinv = 1./d
        self.d_Cov = dinv
        self.Cov = np.dot(U, np.diag(dinv)).dot(U.T)
        self.CovSqrt = np.dot(U, np.diag(np.sqrt(dinv))).dot(U.T)

        self.M = _FiniteDimensionalM(self.dimension)
        self.Msolver = _FiniteDimensionalMsolver(self.dimension)
        self.R = _FiniteDimensionalR(self.CovInv, self.dimension)
        self.Rsolver = _FiniteDimensionalRsolver(self.Cov, self.dimension)

        self.Rd = np.zeros_like(self.mean)

    def cost(self, m):

        if self.regularization:
            d = m - self.mean
            self.R.mult(d, self.Rd)
            return .5*np.dot(self.Rd, d)
        else:
            return 0.

    def grad(self, m, out):

        if self.regularization:
            d = m - self.mean
            self.R.mult(d, out)
        else:
            out[:] = 0.

    # def sample(self, noise, s, add_mean=True):
    #     s[:] = 0.
    #     # print("size noise = ", len(noise.get_local()), "size s = ", len(s.get_local()))
    #     s += self.CovSqrt.dot(noise)
    #
    #     if add_mean:
    #         s += self.mean


if __name__ ==  "__main__":

    import matplotlib.pyplot as plt

    # # # test Brownian motion
    # dimension = 101
    # t = np.linspace(0, 1, dimension)
    #
    # prior = BrownianMotion(dimension)
    #
    # plt.figure()
    # plt.semilogy(prior.d_Cov, '.-')
    # plt.show()
    #
    # noise = np.random.normal(0., 1., dimension)
    # sample = np.zeros(dimension)
    # prior.sample(noise, sample)
    #
    # plt.figure()
    # plt.plot(sample, '.-')
    # plt.title("brownian motion sample")
    # plt.show()
    #
    # Cov = np.zeros((dimension, dimension))
    # for i in range(dimension):
    #     for j in range(dimension):
    #         Cov[i, j] = np.min([t[i], t[j]])
    #
    # d, U = np.linalg.eigh(Cov)
    #
    # CovSqrt = np.dot(U, np.diag(np.sqrt(d))).dot(U.T)
    # dinv = d.copy()
    # dinv[1:] = 1. / d[1:]
    # CovInv = np.dot(U, np.diag(dinv)).dot(U.T)
    #
    # print("CovInv*Cov = ", CovInv.dot(Cov))
    #
    # plt.figure()
    # plt.semilogy(d, ".-")
    # plt.show()

    # # test Laplacian
    dimension = 101
    delta, gamma = 0., 100.
    prior = Laplacian(dimension, delta, gamma)

    plt.figure()
    plt.loglog(prior.d_Cov, '.-')
    plt.show()

    for i in range(10):
        noise = np.random.normal(0., 1., dimension)
        sample = np.zeros(dimension)
        prior.sample(noise, sample)

        plt.subplot(1,2,1)
        plt.plot(sample, '.-')
        plt.subplot(1,2,2)
        alpha = (np.tanh(sample)+1)/2
        plt.plot(alpha, '.-')
        plt.show()
