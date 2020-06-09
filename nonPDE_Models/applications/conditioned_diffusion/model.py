import autograd.numpy as np
from autograd import grad, jacobian, hessian_vector_product

from mpi4py import MPI
comm = MPI.COMM_WORLD

import sys
import os
import pickle

path = "../../"
sys.path.append(path)
from stein import *
from stein.randomizedEigensolver import doublePass

import matplotlib.pyplot as plt


class hvp():
    def __init__(self, hx):

        self.x = None
        self.hx = hx

    def update_x(self, x):

        self.x = x

    def mult(self, xhat, out):

        out[:] = self.hx(self.x, xhat)


class Misfit():
    def __init__(self, dimension, beta, obs=None, loc=None, noise_covariance=None):
        self.dimension = dimension
        self.dt = np.diff(np.linspace(0, 1, dimension+1))
        self.beta = beta
        self.obs = obs
        self.loc = loc
        self.bm = BrownianMotion(dimension)

        self.gx = grad(self.cost)
        self.J = jacobian(self.forward)
        if noise_covariance is not None:
            self.Gamma_noise_inv = np.linalg.inv(noise_covariance)

        self.hx = hessian_vector_product(self.cost)
        self.hvp = hvp(self.hx)

        self.Omega = np.random.normal(0.0, 1.0, (dimension, np.min([100, dimension])+10))

    def solution(self, x):

        x = np.append(0, x)

        dx = np.diff(x)
        u = np.zeros_like(x)
        for i in range(1, len(x)):
            u[i] = u[i - 1] + self.beta * u[i - 1] * (1 - u[i - 1] ** 2) / (1 + u[i - 1] ** 2) * self.dt[i - 1] + dx[i - 1]

        return u

    def forward(self, x):

        x = np.append(0, x)

        dx = np.diff(x)
        u = 0.
        f = np.zeros(1)
        n = len(x)
        for i in range(1, n):
            # u[i] = u[i - 1] + self.beta * u[i - 1] * (1 - u[i - 1] ** 2) / (1 + u[i - 1] ** 2) * self.dt[i - 1] + dx[i - 1]
            u = u + self.beta * u * (1 - u ** 2) / (1 + u ** 2) * self.dt[i - 1] + dx[i - 1]
            f = np.append(f, u)

        u = f[self.loc]

        return u

    def cost(self, x):

        u = self.forward(x)
        diff = (self.obs-u)
        negative_log_likelihood = 0.5 * np.dot(np.dot(diff, self.Gamma_noise_inv), diff)

        return negative_log_likelihood

    def grad(self, x, g):

        g[:] = self.gx(x)

        return g

    def post(self, x):
        negative_log_likelihood = self.cost(x)
        post = np.exp(-np.dot(x, self.bm.CovInv).dot(x)/2. - negative_log_likelihood)

        return post

    def eigdecomp(self, x, k):

        # self.hvp.update_x(x)
        # d, U = doublePass(self.hvp, self.Omega, k, s=1, check=False)

        J = self.J(x)
        if len(np.shape(J)) == 1:
            J = np.array([J])
        Gauss_Newton = np.dot(np.dot(J.T, self.Gamma_noise_inv), J)
        d, U = np.linalg.eigh(Gauss_Newton)

        sort_perm = np.abs(d).argsort()

        sort_perm = sort_perm[::-1]
        d = d[sort_perm[:k]]
        U = U[:, sort_perm[:k]]

        return d, U

    def geom(self, x, geom_ord=[0], k=100):

        loglik = None
        agrad = None
        HessApply = None
        eigs = None

        # get log-likelihood
        if any(s >= 0 for s in geom_ord):
            loglik = - self.cost(x)

        # get gradient
        if any(s >= 1 for s in geom_ord):
            g = np.zeros_like(x)
            agrad = - self.grad(x, g)

        # get Hessian Apply
        if any(s >= 1.5 for s in geom_ord):
            HessApply = None

        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s > 1 for s in geom_ord):
            # eigs = (np.array([1., 0.1]), np.array([np.ones_like(x),-np.ones_like(x)]))
            # eigs = (np.ones(1), np.ones_like(x))
            eigs = self.eigdecomp(x, k=np.min([self.dimension, k]))

        return loglik, agrad, HessApply, eigs


# def solution(t, beta):
#
#     dt = np.diff(t)
#     zero = np.zeros_like(dt)
#     dx = np.random.normal(zero, np.sqrt(dt))
#     u = np.zeros_like(t)
#     x = np.zeros_like(t)
#     for i in range(1, len(t)):
#         u[i] = u[i-1] + beta * u[i-1]*(1-u[i-1]**2)/(1+u[i-1]**2) * dt[i-1] + dx[i-1]
#         x[i] = x[i-1] + dx[i-1]
#
#     return u, x

dimension = 100
prior = BrownianMotion(dimension)

if os.path.isfile("data/observation.p"):
    data = pickle.load(open("data/observation.p", "rb"))
    dimension, beta, x, u, obs, loc, noise_covariance = data
else:
    x_set = []
    for i in range(100):
        sample_noise = np.random.normal(0, 1, dimension)
        sample_noise = comm.bcast(sample_noise, root=0)
        x = np.zeros(dimension)
        prior.sample(sample_noise, x)
        x_set.append(x)
    x_set = np.array(x_set)
    index = np.argmin(x_set[:, -1])

    x = x_set[index, :]

    plt.figure()
    plt.plot(x, '.-')
    plt.show()

    beta = 10.
    misfit = Misfit(dimension, beta)
    u = misfit.solution(x)
    loc = np.arange(0, dimension+1, 5)[1:]
    sigma = 0.1
    obs_noise = np.random.normal(0, sigma, len(loc))
    obs_noise = comm.bcast(obs_noise, root=0)
    obs = u[loc] + obs_noise
    noise_covariance = np.diag(sigma**2*np.ones(len(loc)))

    data = (dimension, beta, x, u, obs, loc, noise_covariance)
    pickle.dump(data, open("data/observation.p", 'wb'))

misfit = Misfit(dimension, beta, obs, loc, noise_covariance)

model = Model(prior, misfit)

if __name__ ==  "__main__":

    dimension = 101
    prior = BrownianMotion(dimension)
    noise = 0.1 + 0.*np.random.normal(0, 1, dimension)
    # noise = np.random.normal(0, 1, dimension)
    x = np.zeros(dimension)
    prior.sample(noise, x)

    beta = 10.
    misfit = Misfit(dimension, beta)
    u = misfit.solution(x)
    loc = np.arange(0, dimension, 5)
    sigma = 0.1
    obs = u[loc] + sigma + 0.*np.random.normal(0, sigma, len(loc))
    noise_covariance = np.diag(sigma**2*np.ones(len(loc)))
    misfit = Misfit(dimension, beta, obs, loc, noise_covariance)

    misfit.x = x
    model = Model(prior, misfit)

    d, U = misfit.eigdecomp(x, k=dimension-1)

    # print("d, U = ", d, U)

    x = np.ones_like(x)
    cost = misfit.cost(x)
    g = np.zeros(dimension)
    gx = misfit.grad(x, g)
    xhat = np.ones(dimension)
    h = np.zeros(dimension)
    misfit.hvp.update_x(x)
    misfit.hvp.mult(xhat, h)
    print("cost, grad = ", cost, g)
    # print("hvp", h)

    mg = np.zeros(dimension)
    model.gradient(x, mg, misfit_only=False)
    # print("model cost, grad = ", model.cost(x), mg)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    # print("rank, nproc, comm", rank, nproc, comm)

    options["number_particles"] = 10
    particle = Particle(model, options, comm)
    # print(particle.particles)

    particle.statistics()

    particle.communication()

    # n = 101
    # x0 = np.linspace(-2, 2, n)
    # x1 = np.linspace(-2, 2, n)
    # cost = np.zeros((n, n))
    #
    # for i in range(n):
    #     for j in range(n):
    #         x = [x0[i], x1[j]]
    #         cost[i, j] = misfit.post(x)
    #
    # x1, x0 = np.meshgrid(x0, x1)
    # plt.figure()
    # plt.contour(x0, x1, cost)
    # plt.show()
    #
