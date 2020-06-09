import autograd.numpy as np
from autograd import grad, jacobian, hessian_vector_product
from autograd.misc import flatten
from autograd.misc.optimizers import adam

from mpi4py import MPI
comm = MPI.COMM_WORLD

import sys
import os
import pickle
import time

path = "../../"
sys.path.append(path)
from stein import *
from stein.randomizedEigensolver import doublePass

import matplotlib.pyplot as plt


def data_rescaling(inputs, outputs):

    for i in range(inputs.shape[1]):
        # inputs[:,i] = (inputs[:,i] - np.min(inputs[:,i]))/(np.max(inputs[:,i])-np.min(inputs[:,i]))
        inputs[:,i] /= 10000

    # for i in range(inputs.shape[1]):
    #     plt.figure()
    #     plt.plot(inputs[:,i], 'r.')
    #     plt.plot(inputs[:,i], '.')
    # plt.show()

    # outputs = np.log(outputs)
    # outputs = (outputs - np.min(outputs))/(np.max(outputs) - np.min(outputs))
    outputs = (outputs + 1) / 2

    # outputs_max = np.max(outputs)
    # outputs_min = np.min(outputs)
    # outputs = -1 + 2 * (outputs - outputs_min) / (outputs_max - outputs_min)

    return inputs, outputs


def check_gradient(flat_args, misfit):
    t0 = time.time()
    # objective_grad = misfit.gx
    # gradient_AD = objective_grad(flat_args)
    gradient_AD = np.zeros_like(flat_args)
    misfit.grad(flat_args, gradient_AD)
    print("time to compute gradient by autograd = ", time.time() - t0, "gradient norm = ", np.sqrt(np.dot(gradient_AD, gradient_AD)))
    # print("gradient by autograd = ", gradient_AD)

    # check gradient by finite difference
    t0 = time.time()
    gradient_FD = []
    eps = 1e-5
    for i in range(len(flat_args)):
        p = flat_args.copy()
        p[i] = flat_args[i] - eps
        obj1 = misfit.cost(p)

        p = flat_args.copy()
        p[i] = flat_args[i] + eps
        obj2 = misfit.cost(p)

        g = (obj2 - obj1) / (2 * eps)
        gradient_FD.append(g)
    print("time to compute gradient by finite difference = ", time.time() - t0)
    # print("gradient by finite difference = ", gradient_FD)

    print("relative error of gradient by FD and AD = ",
          np.linalg.norm(gradient_FD - gradient_AD) / np.linalg.norm(gradient_AD))


class hvp():
    def __init__(self, hx):

        self.x = None
        self.hx = hx

    def update_x(self, x):

        self.x = x

    def mult(self, xhat, out):

        out[:] = self.hx(self.x, xhat)


class Misfit():
    def __init__(self, dimension, inputs, obs, mini_batch=False):
        """These functions implement a standard multi-layer perceptron,
        vectorized over both training examples and weight samples."""
        self.dimension = dimension
        self.prior = FiniteDimensionalPrior(self.dimension)

        self.inputs = inputs
        self.inputs_size = len(inputs)
        self.obs = obs

        self.mini_batch = mini_batch
        if self.mini_batch:
            self.it = 0
            self.mini_batch_size = 32
            self.number_batchs = np.int(np.ceil(self.inputs_size/self.mini_batch_size))

            self.inputs_all = np.copy(inputs)
            self.obs_all = np.copy(obs)

            self.inputs = inputs[:self.mini_batch_size]
            self.obs = obs[:self.mini_batch_size]

        self.gx = grad(self.cost)
        self.J = jacobian(self.forward)

        self.hx = hessian_vector_product(self.cost)
        self.hvp = hvp(self.hx)

    def forward(self, x, inputs=None):
        """Implements a deep neural network.
           params is a list of (weights, bias) tuples.
           inputs is an (N x D) matrix."""

        if inputs is None:
            inputs = self.inputs

        outputs = np.divide(1, 1 + np.exp(-np.dot(inputs, x)))

        # for W, b in x:
        #     outputs = np.dot(inputs, W) + b
        #     # inputs = np.tanh(outputs)
        #     inputs = np.maximum(outputs, 0.)

        return np.reshape(outputs, inputs.shape[0])

    # def unpack_layers(self, weights):
    #
    #     num_weight_sets = len(weights)
    #     for m, n in self.shapes:
    #         yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
    #               weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
    #         weights = weights[:, (m+1)*n:]

    # def forward(self, weights):
    #     """weights is shape (num_weight_samples x num_weights)
    #        inputs  is shape (num_datapoints x D)"""
    #
    #     inputs = np.expand_dims(self.inputs, 0)
    #     for W, b in self.unpack_layers(weights):
    #         outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
    #         inputs = self.nonlinearity(outputs)
    #     return outputs

    def cost(self, x):

        u = self.forward(x)

        negative_log_likelihood = - (np.dot(self.obs, np.log(u)) + np.dot(1-self.obs, np.log(1-u)))

        return negative_log_likelihood

    def grad(self, x, g):

        if self.mini_batch:
            self.it += 1
            self.it = np.mod(self.it, self.number_batchs)
            # print("self.it = ", self.it)
            start = self.it*self.mini_batch_size
            end = np.minimum((self.it+1)*self.mini_batch_size, self.inputs_size)
            self.inputs = self.inputs_all[start:end, :]
            self.obs = self.obs_all[start:end]

        g[:] = self.gx(x)

        # u = self.forward(x)
        # g[:] = 0.
        # for i in range(self.inputs_size):
        #     help = np.exp(-self.inputs[i].dot(x)) * self.inputs[i]
        #     g[:] -= self.obs[i] * np.divide(1, u[i]) * u[i]**2 * help
        #     g[:] += (1-self.obs[i]) * np.divide(1, 1 - u[i]) * help

        return g

    # def hess(self, x, xhat):
    #
    #     return self.hvp(x, xhat)

    def post(self, x):
        negative_log_likelihood = self.cost(x)
        post = np.exp(-np.dot(x, self.prior.CovInv).dot(x)/2. - negative_log_likelihood)

        return post

    def eigdecomp(self, x, k):

        # self.hvp.update_x(x)
        # d, U = doublePass(self.hvp, self.Omega, k, s=1, check=False)

        J = self.J(x)
        if len(np.shape(J)) == 1:
            J = np.array([J])
        Gauss_Newton = np.dot(J.T, J)
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

data = np.load("load_data/cancer.npz")
# data = np.load("load_data/yacht_hydrodynamics.npz")
# data = np.load("load_data/housing.npz")
inputs = data["inputs"]
outputs = data["outputs"]

inputs, outputs = data_rescaling(inputs, outputs)

# number_training = 32  # np.int(len(inputs)*0.9)
# number_testing = 32 # len(inputs) - number_training
number_training = 100
number_testing = len(inputs) - number_training
inputs_training = inputs[:number_training]
inputs_testing = inputs[number_training:number_training+number_testing]
outputs_training = outputs[:number_training]
outputs_testing = outputs[number_training:number_training+number_testing]

dimension = 10000

# plt.figure()
# plt.plot(inputs)
# plt.show()

print("dimension = ", dimension)

if os.path.isfile("data/observation.p"):
    data = pickle.load(open("data/observation.p", "rb"))
    dimension, x, obs = data
    misfit = Misfit(dimension, inputs_training, obs, mini_batch=False)
else:
    # sample_noise = np.random.normal(0, 1, dimension)
    # sample_noise = comm.bcast(sample_noise, root=0)
    # x = np.zeros(dimension)
    # prior.sample(sample_noise, x)

    obs = outputs_training
    misfit = Misfit(dimension, inputs_training, obs, mini_batch=False)

    x0 = np.random.normal(0, 1, dimension)

    from scipy.optimize import minimize

    cost, grad, hessp = misfit.cost(x0), misfit.gx(x0), misfit.hx(x0, x0)
    print("cost, grad, hessp", cost, grad.shape, hessp.shape)

    # plt.figure()
    # plt.plot(grad, '.')
    # plt.show()

    OptRes = minimize(misfit.cost, x0, method='Newton-CG', jac=misfit.gx, hessp=misfit.hx,
                      options={'xtol': 1e-8, 'maxiter': 100, 'disp': True})

    # print("OptRes", OptRes)

    x = OptRes["x"]

    cost, grad, hessp = misfit.cost(x), misfit.gx(x), misfit.hx(x, x)
    print("cost, grad, hessp", cost, grad.shape, hessp.shape)

    # plt.figure()
    # plt.plot(grad, '.')
    # plt.show()

    data = (dimension, x, obs)
    pickle.dump(data, open("data/observation.p", 'wb'))

    # plt.subplot(2, 1, 1)
    # plt.plot(outputs_training, '.')
    # misfit.inputs = inputs_training
    # outputs_forward = misfit.forward(x)
    # # print("outputs_forward = ", outputs_forward)
    # plt.plot(outputs_forward, 'r.')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(outputs_testing, '.')
    # misfit.inputs = inputs_testing
    # outputs_forward = misfit.forward(x)
    # # print("outputs_forward = ", outputs_forward)
    # plt.plot(outputs_forward, 'r.')
    # plt.show()

    # misfit.inputs = inputs_training

# x = [(np.random.randn(m, n),  # weight matrix
#       np.random.randn(n))  # bias vector
#      for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
#
# x, unflatten = flatten(x)
# x0 = x
#
# from scipy.optimize import minimize
# cost, grad, hessp = misfit.cost(x0), misfit.gx(x0), misfit.hx(x0, x0)
# print("cost, grad, hessp", cost, grad.shape, hessp.shape)
# OptRes = minimize(misfit.cost, x0, method='Newton-CG', jac=misfit.gx, hessp=misfit.hx,
#                   options={'xtol': 1e-8, 'maxiter': 100, 'disp': True})
#
# print("OptRes", OptRes)
#
# x = OptRes["x"]

# plt.subplot(2, 1, 1)
# plt.plot(outputs_training, '.')
# misfit.inputs = inputs_training
# outputs_forward = misfit.forward(x)
# # print("outputs_forward = ", outputs_forward)
# plt.plot(outputs_forward, 'r.')
#
# plt.subplot(2, 1, 2)
# plt.plot(outputs_testing, '.')
# misfit.inputs = inputs_testing
# outputs_forward = misfit.forward(x)
# # print("outputs_forward = ", outputs_forward)
# plt.plot(outputs_forward, 'r.')
# plt.show()

# misfit.inputs = inputs_training
sigma = 10 * np.ones_like(x)
prior = FiniteDimensionalPrior(dimension, sigma=sigma, uniform=True)
# prior = FiniteDimensionalPrior(dimension)
prior.CovSqrt = 1 * np.diag(np.ones_like(x))

model = Model(prior, misfit)

if __name__ ==  "__main__":

    test_time = time.time()
    for i in range(100):
        x0 = np.random.normal(0, 1, dimension)
        prior.cost(x)
    print("average cost time = ", (time.time()-test_time)/100)

    # check_gradient(x0, misfit)

    plt.figure()
    plt.plot(x,'.')
    plt.title("optimal weight")
    plt.show()

    plt.figure()
    x0 = np.random.normal(0, 1, dimension)
    plt.plot(misfit.gx(x0),'.')
    plt.title("gradient at random weight")
    # plt.subplot(2,1,2)
    # plt.plot()
    # plt.show()
    plt.show()
    # params = adam(misfit.gx, x, step_size=0.1, num_iters=1000)

    # x = [(np.random.randn(m, n),  # weight matrix
    #       np.random.randn(n))  # bias vector
    #      for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
    #
    # x, unflatten = flatten(x)
    # x0 = x
    #
    # from scipy.optimize import minimize
    #
    # # x0 = np.random.randn(0, 1, dimension)
    # cost, grad, hessp = misfit.cost(x0), misfit.gx(x0), misfit.hx(x0, x0)
    # print("cost, grad, hessp", cost, grad.shape, hessp.shape)
    # OptRes = minimize(misfit.cost, x0, method='Newton-CG', jac=misfit.gx, hessp=misfit.hx,
    #                   options={'xtol': 1e-8, 'maxiter': 100, 'disp': True})
    #
    # print("OptRes", OptRes)

    # x = OptRes["x"]
    x0 = x
    cost, grad, hessp = misfit.cost(x0), misfit.gx(x0), misfit.hx(x0, x0)
    print("cost, grad, hessp", cost, grad.shape, hessp.shape)

    plt.figure()
    plt.plot(grad, '.')
    plt.title("gradient at optimal weight")
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(outputs_training, '.', label="data")
    misfit.inputs = inputs_training
    outputs_forward = misfit.forward(x)
    # print("outputs_forward = ", outputs_forward)
    plt.plot(outputs_forward, 'r.', label="prediction")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(outputs_testing, '.', label="data")
    misfit.inputs = inputs_testing
    outputs_forward = misfit.forward(x)
    # print("outputs_forward = ", outputs_forward)
    plt.plot(outputs_forward, 'r.', label='prediction')
    # plt.show()
    plt.legend()

    outputs_forward = (outputs_forward >= 0.5)
    accuracy = np.sum(np.abs(outputs_testing - outputs_forward) < 1e-3)/len(outputs_testing)
    title = "test accuracy = "+str(accuracy)
    plt.xlabel(title)

    plt.show()

    # plt.figure()
    # plt.plot(outputs, '.')
    # misfit.inputs = inputs
    # outputs_forward = misfit.forward(x)
    # # print("outputs_forward = ", outputs_forward)
    # plt.plot(outputs_forward, 'r.')
    # plt.show()

    # misfit.inputs = inputs_training
    # cost, grad, hessp = misfit.cost(x0), misfit.gx(x0), misfit.hx(x0, x0)
    # print("cost, grad, hessp", cost, grad.shape, hessp.shape)

    # print("x = ", x)
    # d, U = misfit.eigdecomp(flat_args, k=dimension-1)
    #
    # print("d, U = ", d, U)

    # cost = misfit.cost(x)
    # print("cost = ", cost)
    # g = np.zeros(dimension)
    # gx = misfit.grad(x, g)
    # # xhat = np.ones(dimension)
    # # h = np.zeros(dimension)
    # # misfit.hvp.update_x(x)
    # # misfit.hvp.mult(xhat, h)
    # print("grad = ", g)
    # # print("hvp", h)
    # plt.figure()
    # plt.plot(g, '.')
    # plt.show()
    #
    # mg = np.zeros(dimension)
    # model.gradient(x, mg, misfit_only=False)
    # # print("model cost, grad = ", model.cost(x), mg)
    #
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # nproc = comm.Get_size()
    # # print("rank, nproc, comm", rank, nproc, comm)

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
