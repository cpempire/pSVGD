
from __future__ import absolute_import, division, print_function

import numpy as np
from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..modeling.reducedHessian import ReducedHessian
from ..utils.random import Random
from ..algorithms.multivector import MultiVector
from ..algorithms.lowRankOperator import LowRankOperator
from ..algorithms.randomizedEigensolver import doublePass, doublePassG
from mpi4py import MPI

import pickle
import time

plot_valid = False

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except:
    plot_valid = False
    print("can not import pyplot")

if plot_valid:
    import os
    if not os.path.isdir("figure"):
        os.mkdir("figure")


class FisherAverage:
    # to compute the low rank decomposition of the (averaged) Fisher information matrix
    def __init__(self, model, particle, gradient_misfit, comm):
        self.model = model
        self.particle = particle
        self.gradient_misfit = gradient_misfit
        self.comm = comm
        self.nproc = comm.Get_size()

        self.time_communication = 0.
        self.time_computation = 0.

        self.phelp = self.model.generate_vector(PARAMETER)

    def mult(self, phat, pout):

        time_computation = time.time()

        pout.zero()

        for n in range(self.particle.number_particles):
            grad_phat = self.gradient_misfit[n].inner(phat)
            pout.axpy(1.0 / self.particle.number_particles * grad_phat, self.gradient_misfit[n])

        self.time_computation += time.time() - time_computation

        if self.nproc > 1:
            time_communication = time.time()
            pout_array = pout.get_local()
            pout_reduce_array = np.empty([pout_array.size], dtype=float)
            self.comm.Allreduce(pout_array, pout_reduce_array, op=MPI.SUM)
            self.phelp.set_local(pout_reduce_array)
            pout.zero()
            pout.axpy(1/self.nproc, self.phelp)
            self.time_communication += time.time() - time_communication


class FisherAverageProjection:
    # to compute the low rank decomposition of the (averaged) Fisher information matrix with parameter projection
    def __init__(self, model, particle, gradient_misfit, comm):
        self.model = model
        self.particle = particle
        self.gradient_misfit = gradient_misfit
        self.comm = comm
        self.nproc = comm.Get_size()

        self.time_communication = 0.
        self.time_computation = 0.

    def mult(self, phat, pout):

        pout.zero()

        phathelp = self.model.generate_vector(PARAMETER)
        phat_array = phat.get_local()
        for r in range(self.particle.coefficient_dimension):
            phathelp.axpy(phat_array[r], self.particle.bases[r])

        pouthelp = self.model.generate_vector(PARAMETER)
        for n in range(self.particle.number_particles):
            grad_phat = self.gradient_misfit[n].inner(phathelp)
            pouthelp.axpy(1.0 / self.particle.number_particles * grad_phat, self.gradient_misfit[n])

        pout_array = np.empty(self.particle.coefficient_dimension, dtype=float)
        for r in range(self.particle.coefficient_dimension):
            pout_array[r] = self.particle.bases[r].inner(pouthelp)
        pout.set_local(pout_array)

        if self.nproc > 1:
            pout_reduce_array = np.empty(pout_array.size, dtype=float)
            self.comm.Allreduce(pout_array, pout_reduce_array, op=MPI.SUM)
            pout_reduce_array /= self.nproc
            pout.set_local(pout_reduce_array)


class HessianAverage:
    # to compute the low rank decomposition of the averaged Hessian
    def __init__(self, model, particle, options, x_all, comm, d, U):
        self.model = model
        self.particle = particle
        self.x_all = x_all
        self.gauss_newton_approx = options["gauss_newton_approx"]
        self.comm = comm
        self.nproc = comm.Get_size()
        self.d = d
        self.U = U

        self.time_communication = 0.
        self.time_computation = 0.

    def mult(self, phat, pout):

        time_computation = time.time()

        pout.zero()

        for n in range(self.particle.number_particles):
            phelp = self.model.generate_vector(PARAMETER)

            # self.model.setPointForHessianEvaluations(self.x_all[n], self.gauss_newton_approx)
            # hessian_misfit = ReducedHessian(self.model, misfit_only=True)

            if self.d[0] is None:
                self.model.setPointForHessianEvaluations(self.x_all[n], self.gauss_newton_approx)
                hessian_misfit = ReducedHessian(self.model, misfit_only=True)
            else:
                hessian_misfit = LowRankOperator(self.d[n], self.U[n])

            hessian_misfit.mult(phat, phelp)
            pout.axpy(1.0 / self.particle.number_particles, phelp)

        self.time_computation += time.time() - time_computation

        if self.nproc > 1:
            time_communication = time.time()
            pout_array = pout.get_local()
            pout_reduce_array = np.empty([pout_array.size], dtype=float)
            self.comm.Allreduce(pout_array, pout_reduce_array, op=MPI.SUM)
            phelp = self.model.generate_vector(PARAMETER)
            phelp.set_local(pout_reduce_array)
            pout.zero()
            pout.axpy(1/self.nproc, phelp)
            self.time_communication += time.time() - time_communication


class ReducedHessianProjection:
    # to compute the low rank decomposition of the averaged Hessian with parameter projection
    def __init__(self, model, particle, hessian_misfit):
        self.model = model
        self.particle = particle
        self.hessian_misfit = hessian_misfit

    def mult(self, phat, pout):

        pout.zero()

        phathelp = self.model.generate_vector(PARAMETER)
        phat_array = phat.get_local()
        for r in range(self.particle.coefficient_dimension):
            phathelp.axpy(phat_array[r], self.particle.bases[r])

        phelp = self.model.generate_vector(PARAMETER)
        self.hessian_misfit.mult(phathelp, phelp)

        pout_array = np.empty(self.particle.coefficient_dimension, dtype=float)
        for r in range(self.particle.coefficient_dimension):
            pout_array[r] = self.particle.bases[r].inner(phelp)
        pout.set_local(pout_array)


class HessianAverageProjection:
    # to compute the low rank decomposition of the averaged Hessian with parameter projection
    def __init__(self, model, particle, options, x_all, comm, d, U):
        self.model = model
        self.particle = particle
        self.x_all = x_all
        self.gauss_newton_approx = options["gauss_newton_approx"]
        self.comm = comm
        self.nproc = comm.Get_size()
        self.d = d
        self.U = U

    def mult(self, phat, pout):

        pout.zero()

        phathelp = self.model.generate_vector(PARAMETER)
        phat_array = phat.get_local()
        for r in range(self.particle.coefficient_dimension):
            phathelp.axpy(phat_array[r], self.particle.bases[r])

        pouthelp = self.model.generate_vector(PARAMETER)
        for n in range(self.particle.number_particles):
            phelp = self.model.generate_vector(PARAMETER)
            if self.d[0] is None:
                self.model.setPointForHessianEvaluations(self.x_all[n], self.gauss_newton_approx)
                hessian_misfit = ReducedHessian(self.model, misfit_only=True)
            else:
                hessian_misfit = LowRankOperator(self.d[n], self.U[n])
            hessian_misfit.mult(phat, phelp)
            pouthelp.axpy(1.0 / self.particle.number_particles, phelp)

        pout_array = np.empty(self.particle.coefficient_dimension, dtype=float)
        for r in range(self.particle.coefficient_dimension):
            pout_array[r] = self.particle.bases[r].inner(pouthelp)
        pout.set_local(pout_array)

        if self.nproc > 1:
            pout_reduce_array = np.empty(pout_array.size, dtype=float)
            self.comm.Allreduce(pout_array, pout_reduce_array, op=MPI.SUM)
            pout_reduce_array /= self.nproc
            pout.set_local(pout_reduce_array)


class Variation:
    # compute the variation (gradient, Hessian) of the negative log likelihood function
    def __init__(self, model, particle, options, comm):
        # if add particles, then use generalized eigenvalue decomposition
        if options["add_number"] > 0:
            options["low_rank_Hessian"] = 2

        self.model = model  # forward model
        self.particle = particle  # particle class
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.type_approximation = options["type_approximation"]
        self.low_rank_Hessian = options["low_rank_Hessian"]
        self.rank_Hessian = options["rank_Hessian"]
        self.rank_Hessian_tol = options["rank_Hessian_tol"]
        self.low_rank_Hessian_average = options["low_rank_Hessian_average"]
        self.rank_Hessian_average = options["rank_Hessian_average"]
        self.rank_Hessian_average_tol = options["rank_Hessian_average_tol"]
        self.low_rank_Hessian_hold = options["low_rank_Hessian"]
        self.low_rank_Hessian_average_hold = options["low_rank_Hessian_average"]
        self.gauss_newton_approx = options["gauss_newton_approx"]
        self.gauss_newton_approx_hold = options["gauss_newton_approx"]
        self.max_iter_gauss_newton_approx = options["max_iter_gauss_newton_approx"]
        self.is_bases_updated = False

        if options["add_number"] > 0:
            self.low_rank_Hessian = True  # use generalized eigendecomposition if new particles are to be added
            self.low_rank_Hessian_hold = True
            self.low_rank_Hessian_average = True  # use generalized eigendecomposition if new particles are to be added
            self.low_rank_Hessian_hold_average = True

        self.randomGen = Random(seed=1)

        if self.options["is_projection"]:
            vector = self.model.generate_vector(PARAMETER)
            nvec = self.rank_Hessian_average + 5
            self.Omega_average = self.generate_Omega(vector, nvec)

            if self.low_rank_Hessian:
                vector = self.particle.generate_vector()
                nvec = self.rank_Hessian + 5
                self.Omega_projection = self.generate_Omega(vector, nvec)

            if self.low_rank_Hessian_average:
                vector = self.particle.generate_vector()
                nvec = self.rank_Hessian_average + 5
                self.Omega_average_projection = self.generate_Omega(vector, nvec)
        else:
            if self.low_rank_Hessian:
                vector = self.model.generate_vector(PARAMETER)
                nvec = self.rank_Hessian + 5
                self.Omega = self.generate_Omega(vector, nvec)

            if self.low_rank_Hessian_average:
                vector = self.model.generate_vector(PARAMETER)
                nvec = self.rank_Hessian_average + 5
                self.Omega_average = self.generate_Omega(vector, nvec)

        self.iter_old = 0

        self.x_all = [None] * self.particle.number_particles_all
        for n in range(self.particle.number_particles_all):
            self.x_all[n] = [self.model.generate_vector(STATE), self.model.generate_vector(PARAMETER),
                             self.model.generate_vector(ADJOINT)]
        self.x_all_gather = None

        self.grad_norm = np.empty(self.particle.number_particles_all, dtype=float)
        self.grad = [self.particle.generate_vector() for n in range(self.particle.number_particles_all)]
        self.gradient_gather = None
        self.gradient_norm_gather = None
        self.d = [None] * self.particle.number_particles_all
        self.d_gather = None
        self.U = [None] * self.particle.number_particles_all
        self.U_gather = None
        self.d_average = None
        self.d_average_save = None
        self.U_average = None
        self.grad = None
        self.gradient_misfit = None
        self.hessian_misfit = None
        if self.options["is_projection"]:
            self.hessian_misfit_projected = None
            self.hessian_misfit_gather = None
            self.hessian_misfit_average = None
            self.fisher_misfit_average = None

        self.time_communication = 0.
        self.time_computation = 0.
        self.time_update_bases_communication = 0.
        self.time_update_bases_computation = 0.
        self.number_update_bases = 0

    def initialization(self):
        self.x_all = [None] * self.particle.number_particles_all
        for n in range(self.particle.number_particles_all):
            self.x_all[n] = [self.model.generate_vector(STATE), self.model.generate_vector(PARAMETER),
                             self.model.generate_vector(ADJOINT)]
        self.x_all_gather = None

        self.grad_norm = np.empty(self.particle.number_particles_all, dtype=float)
        self.grad = [self.particle.generate_vector() for n in range(self.particle.number_particles_all)]
        self.gradient_gather = None
        self.gradient_norm_gather = None
        self.d = [None] * self.particle.number_particles_all
        self.d_gather = None
        self.U = [None] * self.particle.number_particles_all
        self.U_gather = None
        self.d_average = None
        self.d_average_save = None
        self.U_average = None
        self.grad = None
        self.gradient_misfit = None
        self.hessian_misfit = None
        if self.options["is_projection"]:
            self.hessian_misfit_projected = None
            self.hessian_misfit_gather = None
            self.hessian_misfit_average = None
            self.fisher_misfit_average = None

    def generate_Omega(self, vector, nvec):
        # generate (the same accross processors) Gaussian random vectors to be used for randomized SVD
        Omega = MultiVector(vector, nvec)
        self.randomGen.normal(1., Omega)

        return Omega

    def communication(self):  # gather solutions, gradients and eigenpairs from each processor to all processors
        time_communication = time.time()

        # gather gradients
        gradient_array = np.empty([self.particle.number_particles_all, self.particle.dimension], dtype=float)
        gradient_gather_array = np.empty([self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
        for n in range(self.particle.number_particles_all):
            gradient_array[n, :] = self.grad[n].get_local()
        self.comm.Allgather(gradient_array, gradient_gather_array)

        self.gradient_gather = [[self.particle.generate_vector() for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
        for p in range(self.nproc):
            for n in range(self.particle.number_particles_all):
                self.gradient_gather[p][n].set_local(gradient_gather_array[p, n, :])

        # norm of gradients
        self.gradient_norm_gather = np.empty([self.nproc, self.particle.number_particles_all], dtype=float)
        self.comm.Allgather(self.grad_norm, self.gradient_norm_gather)

        if self.options["type_approximation"] is "hessian":
            # gather Hessian
            if self.options["is_projection"]:
                hess_gather_array = np.empty([self.nproc, self.particle.number_particles_all,
                                              self.particle.coefficient_dimension, self.particle.coefficient_dimension], dtype=float)
                self.comm.Allgather(self.hessian_misfit_projected, hess_gather_array)
                self.hessian_misfit_gather = hess_gather_array

            elif self.low_rank_Hessian:
                rank_Hessian = self.rank_Hessian

                # eigenvalues
                d_array = np.empty([self.particle.number_particles_all, rank_Hessian], dtype=float)
                d_gather_array = np.empty([self.nproc, self.particle.number_particles_all, rank_Hessian], dtype=float)
                for n in range(self.particle.number_particles_all):
                        d_array[n, :] = self.d[n]
                self.comm.Allgather(d_array, d_gather_array)

                self.d_gather = [[None for n in range(self.particle.number_particles_all)] for p in  range(self.nproc)]
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        self.d_gather[p][n] = d_gather_array[p, n, :]

                # eigenvectors
                U_array = np.empty([self.particle.number_particles_all, rank_Hessian, self.particle.dimension], dtype=float)
                U_gather_array = np.empty([self.nproc, self.particle.number_particles_all, rank_Hessian, self.particle.dimension], dtype=float)
                for n in range(self.particle.number_particles_all):
                    for r in range(rank_Hessian):
                        U_array[n, r, :] = self.U[n][r].get_local()
                self.comm.Allgather(U_array, U_gather_array)

                if self.options["print_level"] > 2:
                    print("size of p ("+str(self.nproc)+") x n ("+str(self.particle.number_particles_all)+") x r ("+str(rank_Hessian)+")"
                          " eigenvectors ("+str(self.particle.particle_dimension)+") = ", '{:.2e}'.format(self.nproc*U_gather_array[0].nbytes), " bytes")

                self.U_gather = [[MultiVector(self.particle.generate_vector(), rank_Hessian) for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        for r in range(rank_Hessian):
                            self.U_gather[p][n][r].set_local(U_gather_array[p, n, r, :])

            else:
                # gather solutions
                x_all_gather_array = [None] * 3
                for i in [STATE, PARAMETER, ADJOINT]:
                    x_all_array = np.empty([self.particle.number_particles_all, self.x_all[0][i].get_local().size],
                                           dtype=float)
                    x_all_gather_array[i] = np.empty(
                        [self.nproc, self.particle.number_particles_all, self.x_all[0][i].get_local().size], dtype=float)
                    for n in range(self.particle.number_particles_all):
                        x_all_array[n, :] = self.x_all[n][i].get_local()
                    self.comm.Allgather(x_all_array, x_all_gather_array[i])

                self.x_all_gather = [[None for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        self.x_all_gather[p][n] = [self.model.generate_vector(STATE),
                                                   self.model.generate_vector(PARAMETER),
                                                   self.model.generate_vector(ADJOINT)]
                        [self.x_all_gather[p][n][i].set_local(x_all_gather_array[i][p, n, :]) for i in
                         [STATE, PARAMETER, ADJOINT]]

        self.time_communication += time.time() - time_communication

    def update_bases(self, particle):
        self.number_update_bases += 1

        t0 = time.time()
        if self.options["type_projection"] is "hessian":
            hessian_misfit = HessianAverage(self.model, particle, self.options, self.x_all, self.comm, self.d, self.U)
        elif self.options["type_projection"] is "fisher":
            hessian_misfit = FisherAverage(self.model, particle, self.gradient_misfit, self.comm)
        else:
            raise NotImplementedError("choose type_projection as hessian or fisher")

        self.d_average, self.U_average = doublePassG(hessian_misfit, self.model.prior.R,
                                                     self.model.prior.Rsolver,
                                                     self.Omega_average, self.rank_Hessian_average, s=1)
        t1 = time.time() - t0
        if self.options["print_level"] > 2:
            print("time to compute low (" + str(self.rank_Hessian_average) + ") rank decompositions of hessian_misfit = ", t1)
            if self.rank == 0:
                print("\n{0:4} {1:10}".format("i", "eigenvalue"))
                for m in range(len(self.d_average)):
                    print("{0:3d} {1:10e}".format(m, self.d_average[m]))

        self.time_update_bases_communication += hessian_misfit.time_communication
        self.time_update_bases_computation += hessian_misfit.time_computation

        # update ranks according to tolerance
        rank_Hessian_average = np.argmax(self.d_average < self.rank_Hessian_average_tol)
        if rank_Hessian_average == 0:
            rank_Hessian_average = self.rank_Hessian_average

        if rank_Hessian_average < self.rank_Hessian_average:
            self.rank_Hessian_average = np.max([rank_Hessian_average, 1])
            self.d_average = self.d_average[:rank_Hessian_average]
            U_average = MultiVector(self.model.generate_vector(PARAMETER), rank_Hessian_average)
            Omega_average = MultiVector(self.model.generate_vector(PARAMETER), rank_Hessian_average + 5)
            for r in range(rank_Hessian_average):
                U_average[r].axpy(1.0, self.U_average[r])
            self.U_average = U_average
            for r in range(rank_Hessian_average + 5):
                Omega_average[r].axpy(1.0, self.Omega_average[r])
            self.Omega_average = Omega_average

        self.d_average_save = self.d_average

        # perform the projection
        particle.update_bases(self.U_average)
        self.particle = particle

    def update(self, particle, iter=0, step_norm=1):
        self.particle = particle

        if particle.number_particles_all > particle.number_particles_all_old:
            self.initialization()

        time_computation = time.time()
        t0 = time.time()
        # solve the forward and adjoint problems, and evaluate gradient at all the particles
        for n in range(particle.number_particles_all):
            # solve the state and adjoint problems
            self.x_all[n][PARAMETER].zero()
            self.x_all[n][PARAMETER].axpy(1.0, particle.particles[n])
            self.model.solveFwd(self.x_all[n][STATE], self.x_all[n])
            self.model.solveAdj(self.x_all[n][ADJOINT], self.x_all[n])
        t1 = time.time() - t0
        if self.options["print_level"] > 2:
            print("time to solve " + str(particle.number_particles_all) + " fwd and adj problems = ", t1)

        if self.options["type_projection"] is "fisher" or self.options["type_approximation"] is "fisher":
            self.gradient_misfit = [self.model.generate_vector(PARAMETER) for n in range(particle.number_particles)]
            for n in range(particle.number_particles):
                self.model.evalGradientParameter(self.x_all[n], self.gradient_misfit[n], misfit_only=True)

        if self.options["is_projection"]:
            # add conditions to update bases
            self.is_bases_updated = False
            if (iter == 0) or (step_norm < self.options["step_projection_tolerance"]):  # np.mod(iter, particle.coefficient_dimension)
                self.update_bases(particle)
                particle.update_dimension(self.d_average)
                self.particle = particle

                if self.rank == 0:
                    print("updated projection bases at iter " + str(iter) + " with step norm = " + str(step_norm)
                          + " and dimension of the subspace = "+str(particle.coefficient_dimension))

                if iter - self.iter_old == 1:
                    self.options["step_projection_tolerance"] /= self.options["reduce_step_projection_tolerance"]

                self.iter_old = iter
                self.is_bases_updated = True

            # gradient
            self.grad = [particle.generate_vector() for n in range(particle.number_particles_all)]
            for n in range(particle.number_particles_all):
                gradient = self.model.generate_vector(PARAMETER)
                self.model.evalGradientParameter(self.x_all[n], gradient)
                grad = np.empty(particle.coefficient_dimension, dtype=float)
                for r in range(particle.coefficient_dimension):
                    grad[r] = gradient.inner(particle.bases[r])
                self.grad[n].set_local(grad)
                self.grad_norm[n] = np.sqrt(self.grad[n].inner(self.grad[n]))

            # Hessian for Newton-based optimization and kernel construction
            t0 = time.time()
            if self.options["type_approximation"] is "fisher":
                hessian_misfit = FisherAverageProjection(self.model, particle, self.gradient_misfit, self.comm)
                fisher = np.empty([particle.coefficient_dimension, particle.coefficient_dimension], dtype=float)
                for r in range(self.particle.coefficient_dimension):
                    delta = np.zeros(self.particle.coefficient_dimension)
                    delta[r] = 1.0
                    pout = self.particle.generate_vector()
                    phat = self.particle.generate_vector()
                    phat.set_local(delta)
                    hessian_misfit.mult(phat, pout)
                    fisher[r, :] = pout.get_local()

                d, U = np.linalg.eig(fisher)
                d = np.multiply(np.sqrt(np.abs(d)), np.sign(d))  # square root of eigenvalues
                fisher = np.dot(U, np.diag(d)).dot(U.T)

                self.fisher_misfit_average = fisher
                self.hessian_misfit_gather = np.empty([self.nproc, self.particle.number_particles_all,
                                              self.particle.coefficient_dimension, self.particle.coefficient_dimension],
                                             dtype=float)
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        self.hessian_misfit_gather[p][n] = fisher
            else:
                # assemble the full Hessian misfit matrix of r x r by r Hessian action on Kronecker delta vector
                hess = np.empty([particle.number_particles_all, particle.coefficient_dimension, particle.coefficient_dimension], dtype=float)
                for n in range(particle.number_particles_all):
                    self.model.setPointForHessianEvaluations(self.x_all[n], self.gauss_newton_approx)
                    hessian_misfit = ReducedHessian(self.model, misfit_only=True)
                    hessian_misfit = ReducedHessianProjection(self.model, self.particle, hessian_misfit)
                    for r in range(self.particle.coefficient_dimension):
                        delta = np.zeros(self.particle.coefficient_dimension)
                        delta[r] = 1.0
                        pout = self.particle.generate_vector()
                        phat = self.particle.generate_vector()
                        phat.set_local(delta)
                        hessian_misfit.mult(phat, pout)
                        hess[n, r, :] = pout.get_local()
                self.hessian_misfit_projected = hess

                hess_average = np.mean(hess, axis=0).astype(float)
                hessian_misfit_average = np.empty([particle.coefficient_dimension, particle.coefficient_dimension], dtype=float)
                self.comm.Allreduce(hess_average, hessian_misfit_average, op=MPI.SUM)
                self.hessian_misfit_average = np.divide(hessian_misfit_average, self.nproc)

            t1 = time.time() - t0
            if self.options["print_level"] > 2:
                print("time to compute " + str(particle.number_particles_all) + " projected Hessian = ", t1)

        else:
            # gradient
            self.grad = [self.model.generate_vector(PARAMETER) for n in range(self.particle.number_particles_all)]
            for n in range(particle.number_particles_all):
                self.grad_norm[n] = self.model.evalGradientParameter(self.x_all[n], self.grad[n])

            # Hessian for Newton-based optimization and kernel construction
            if self.options["type_approximation"] is "fisher":
                if self.low_rank_Hessian or self.low_rank_Hessian_average:
                    t0 = time.time()
                    fisher_misfit = FisherAverage(self.model, particle, self.gradient_misfit, self.comm)
                    self.d_average, self.U_average = doublePassG(fisher_misfit, self.model.prior.R, self.model.prior.Rsolver,
                                                               self.Omega_average, self.rank_Hessian_average, s=1)
                    t1 = time.time() - t0
                    if self.options["print_level"] > 2:
                        print("time to compute " + str(particle.number_particles_all) + " low (" + str(
                            self.rank_Hessian) + ") rank decompositions = ", t1)
                    self.d_average_save = self.d_average
                    self.d_average = np.multiply(np.sqrt(np.abs(self.d_average)), np.sign(self.d_average))

                    # update ranks according to tolerance
                    rank_Hessian_average = np.argmax(self.d_average < self.rank_Hessian_average_tol)
                    if rank_Hessian_average == 0:
                        rank_Hessian_average = self.rank_Hessian_average

                    if rank_Hessian_average < self.rank_Hessian_average:
                        self.rank_Hessian_average = np.max([rank_Hessian_average, 1])
                        self.d_average = self.d_average[:rank_Hessian_average]
                        U_average = MultiVector(self.model.generate_vector(PARAMETER), rank_Hessian_average)
                        Omega_average = MultiVector(self.model.generate_vector(PARAMETER), rank_Hessian_average+5)
                        for r in range(rank_Hessian_average):
                            U_average[r].axpy(1.0, self.U_average[r])
                        self.U_average = U_average
                        for r in range(rank_Hessian_average+5):
                            Omega_average[r].axpy(1.0, self.Omega_average[r])
                        self.Omega_average = Omega_average

            else:
                if self.low_rank_Hessian:
                    t0 = time.time()
                    rank_max = np.empty(particle.number_particles_all, dtype=int)
                    # compute the low rank decomposition only for the particles used in constructing the transport map
                    for n in range(particle.number_particles_all):
                        # low rank decomposition of hessian misfit
                        self.model.setPointForHessianEvaluations(self.x_all[n], self.gauss_newton_approx)
                        hessian_misfit = ReducedHessian(self.model, misfit_only=True)
                        self.d[n], self.U[n] = doublePassG(hessian_misfit, self.model.prior.R, self.model.prior.Rsolver,
                                                           self.Omega, self.rank_Hessian, s=1)
                        rank_max[n] = np.argmax(self.d[n] < self.rank_Hessian_tol)
                    t1 = time.time() - t0
                    if self.options["print_level"] > 2:
                        print("time to compute " + str(particle.number_particles_all) + " low (" + str(self.rank_Hessian) + ") rank decompositions = ", t1)

                    self.rank_max = np.max(rank_max)
                    if self.rank_max == 0:
                        self.rank_max = np.array(int(self.rank_Hessian))

                    # update maximum rank across all processors
                    rank_max = np.empty([self.nproc], dtype=int)
                    self.comm.Allgather(self.rank_max, rank_max)
                    self.rank_max = np.max(rank_max)

                    # update rank of the Hessian at each particle according to the tolerance
                    if self.rank_max == 0:
                        self.rank_max = self.rank_Hessian
                    if self.rank_max < self.rank_Hessian:
                        self.rank_Hessian = self.rank_max
                        for n in range(particle.number_particles_all):
                            self.d[n] = self.d[n][:self.rank_max]
                            U = MultiVector(self.model.generate_vector(PARAMETER), self.rank_max)
                            for r in range(self.rank_max):
                                U[r].axpy(1.0, self.U[n][r])
                            self.U[n] = U
                        Omega = MultiVector(self.model.generate_vector(PARAMETER), self.rank_max + 5)
                        for r in range(self.rank_max + 5):
                            Omega[r].axpy(1.0, self.Omega[r])
                        self.Omega = Omega

                # metric
                if self.low_rank_Hessian_average:
                    t0 = time.time()
                    hessian_misfit = HessianAverage(self.model, particle, self.options, self.x_all, self.comm, self.d, self.U)
                    self.d_average, self.U_average = doublePassG(hessian_misfit, self.model.prior.R,
                                                                 self.model.prior.Rsolver,
                                                                 self.Omega_average, self.rank_Hessian_average, s=1)
                    # update ranks according to tolerance
                    rank_Hessian_average = np.argmax(self.d_average < self.rank_Hessian_average_tol)
                    if rank_Hessian_average == 0:
                        rank_Hessian_average = self.rank_Hessian_average

                    if rank_Hessian_average < self.rank_Hessian_average:
                        self.rank_Hessian_average = np.max([rank_Hessian_average, 1])
                        self.d_average = self.d_average[:rank_Hessian_average]
                        U_average = MultiVector(self.model.generate_vector(PARAMETER), rank_Hessian_average)
                        Omega_average = MultiVector(self.model.generate_vector(PARAMETER), rank_Hessian_average+5)
                        for r in range(rank_Hessian_average):
                            U_average[r].axpy(1.0, self.U_average[r])
                        self.U_average = U_average
                        for r in range(rank_Hessian_average+5):
                            Omega_average[r].axpy(1.0, self.Omega_average[r])
                        self.Omega_average = Omega_average

                    t1 = time.time() - t0
                    if self.options["print_level"] > 2:
                        print("time to compute low rank decompositions of averaged Hessian/Fisher = ", t1)
                        if self.rank == 0:
                            print("\n{0:4} {1:10}".format("i", "eigenvalue"))
                            for m in range(len(self.d_average)):
                                print("{0:3d} {1:10e}".format(m, self.d_average[m]))

                    self.d_average_save = self.d_average

        self.time_computation += time.time() - time_computation

        t0 = time.time()
        self.communication()
        t1 = time.time()-t0
        if self.options["print_level"] > 2:
            print("time to perform communication for variation data in " + str(self.nproc) + " processors = ", t1)

    def qoi_statistics(self):
        qoi_tmp = np.zeros(self.particle.number_particles_all)
        for n in range(self.particle.number_particles_all):
            qoi_tmp[n] = self.model.qoi.eval(self.x_all[n])

        qoi = np.zeros((self.nproc, self.particle.number_particles_all))
        self.comm.Allgather(qoi_tmp, qoi)

        mean = np.mean(qoi)
        std = np.std(qoi)
        moment2 = np.mean(qoi**2)

        return mean, std, moment2

    def gradient(self, pout, n):
        self.model.evalGradientParameter(self.x_all[n], pout, misfit_only=False)

    def hessian(self, phat, pout, n, p=None):
        # misfit Hessian action at particle pn in direction phat , H * phat = pout
        if self.options["type_approximation"] is "fisher":
            if self.options["is_projection"]:
                pout.set_local(self.fisher_misfit_average.dot(phat.get_local()))
            else:
                if self.low_rank_Hessian:
                    fisher_misfit = LowRankOperator(self.d_average, self.U_average)
                else:
                    fisher_misfit = FisherAverage(self.model, self.particle, self.gradient_misfit, self.comm)

                fisher_misfit.mult(phat, pout)
        else:
            if self.options["is_projection"]:
                if p is None:
                    pout.set_local(self.hessian_misfit_projected[n].dot(phat.get_local()))
                else:
                    pout.set_local(self.hessian_misfit_gather[p][n].dot(phat.get_local()))
            else:
                if p is None:
                    if self.low_rank_Hessian:
                        hessian_misfit = LowRankOperator(self.d[n], self.U[n])
                    else:
                        self.model.setPointForHessianEvaluations(self.x_all[n], self.gauss_newton_approx)
                        hessian_misfit = ReducedHessian(self.model, misfit_only=True)
                else:
                    if self.low_rank_Hessian:
                        hessian_misfit = LowRankOperator(self.d_gather[p][n], self.U_gather[p][n])
                    else:
                        self.model.setPointForHessianEvaluations(self.x_all_gather[p][n], self.gauss_newton_approx)
                        hessian_misfit = ReducedHessian(self.model, misfit_only=True)

                hessian_misfit.mult(phat, pout)

    def hessian_average(self, phat, pout):
        if self.options["type_approximation"] is 'fisher':
            if self.options["is_projection"]:
                pout.set_local(self.fisher_misfit_average.dot(phat.get_local()))
            else:
                if self.low_rank_Hessian_average:
                    hessian_misfit = LowRankOperator(self.d_average, self.U_average)
                else:
                    hessian_misfit = FisherAverage(self.model, self.particle, self.gradient_misfit, self.comm)
                hessian_misfit.mult(phat, pout)
        else:
            if self.options["is_projection"]:
                pout.set_local(self.hessian_misfit_average.dot(phat.get_local()))
            else:
                if self.low_rank_Hessian_average:
                    hessian_misfit = LowRankOperator(self.d_average, self.U_average)
                else:
                    hessian_misfit = HessianAverage(self.model, self.particle, self.options, self.x_all, self.comm, self.d, self.U)
                hessian_misfit.mult(phat, pout)

    def save_eigenvalue(self, save_number=1, it=0):
        # for n in range(save_number):
        #     filename = "data/rank_" + str(self.rank) + "_eigenvalue_" + str(n) + '_iteration_' + str(it) + '.p'
        #     pickle.dump(self.d, open(filename, 'wb'))
        if False:
            if self.options["is_projection"] or self.low_rank_Hessian_average:
                filename = "data/d_average_nDimension_"+str(self.particle.particle_dimension)+"_nCore_"+str(self.nproc)+\
                           "_nSamples_"+str(self.nproc*self.particle.number_particles_all)+\
                           "_nData_"+str(len(self.model.misfit.d))+'_iteration_'+str(it)+".p"
                pickle.dump(self.d_average_save, open(filename, 'wb'))

    def plot_eigenvalue(self, save_number=1, it=0):

        if plot_valid and self.low_rank_Hessian:

            for n in range(save_number):
                if n < self.particle.number_particles_all and self.d[n] is not None:
                    d = self.d[n]

                    fig = plt.figure()
                    if np.all(d > 0):
                        plt.semilogy(d, 'r.')
                    else:
                        indexplus = np.where(d > 0)[0]
                        # print indexplus
                        dplus, = plt.semilogy(indexplus, d[indexplus], 'ro')
                        indexminus = np.where(d < 0)[0]
                        # print indexminus
                        dminus, = plt.semilogy(indexminus, -d[indexminus], 'k*')

                        plt.legend([dplus, dminus], ["positive", "negative"])

                    plt.xlabel("n ", fontsize=12)
                    plt.ylabel("|$\lambda_n$|", fontsize=12)

                    plt.tick_params(axis='both', which='major', labelsize=12)
                    plt.tick_params(axis='both', which='minor', labelsize=12)

                    filename = "figure/rank_" + str(self.rank) + "_eigenvalue_" + str(n) + '_iteration_' + str(it) + '.pdf'
                    fig.savefig(filename, format='pdf')
                    filename = "figure/rank_" + str(self.rank) + "_eigenvalue_" + str(n) + '_iteration_' + str(it) + '.eps'
                    fig.savefig(filename, format='eps')

                    plt.close()

        if plot_valid and (self.low_rank_Hessian_average or self.options["is_projection"]) and self.d_average_save is not None:
            d = self.d_average_save
            fig = plt.figure()
            if np.all(d > 0):
                plt.semilogy(d, 'r.')
            else:
                indexplus = np.where(d > 0)[0]
                # print indexplus
                dplus, = plt.semilogy(indexplus, d[indexplus], 'ro')
                indexminus = np.where(d < 0)[0]
                # print indexminus
                dminus, = plt.semilogy(indexminus, -d[indexminus], 'k*')

                plt.legend([dplus, dminus], ["positive", "negative"])

            plt.xlabel("n ", fontsize=12)
            plt.ylabel("|$\lambda_n$|", fontsize=12)

            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tick_params(axis='both', which='minor', labelsize=12)

            filename = "figure/average_eigenvalue" + '_iteration_' + str(it) + '.pdf'
            fig.savefig(filename, format='pdf')
            filename = "figure/average_eigenvalue" + '_iteration_' + str(it) + '.eps'
            fig.savefig(filename, format='eps')

        if plot_valid and it == 0:
            r = self.particle.particle_dimension
            if r > 100:
                r = 100
            randomGen = Random(seed=1)
            Omega = MultiVector(self.model.generate_vector(PARAMETER), r + 5)
            for i in range(r + 5):
                randomGen.normal(1., Omega[i])

            from ..algorithms.linalg import Solver2Operator, Operator2Solver
            d, _ = doublePassG(Solver2Operator(self.model.prior.Rsolver),
                               Solver2Operator(self.model.prior.Msolver),
                               Operator2Solver(self.model.prior.M),
                               Omega, r, s = 1, check = False )

            # d, U = doublePass(self.model.prior.Rsolver, Omega, r, s=1)

            fig = plt.figure()
            if np.all(d > 0):
                plt.semilogy(d, 'r.')
            else:
                indexplus = np.where(d > 0)[0]
                # print indexplus
                dplus, = plt.semilogy(indexplus, d[indexplus], 'ro')
                indexminus = np.where(d < 0)[0]
                # print indexminus
                dminus, = plt.semilogy(indexminus, -d[indexminus], 'k*')

                plt.legend([dplus, dminus], ["positive", "negative"])

            plt.xlabel("n ", fontsize=12)
            plt.ylabel("|$\lambda_n$|", fontsize=12)

            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tick_params(axis='both', which='minor', labelsize=12)

            filename = "figure/prior_eigenvalue.pdf"
            fig.savefig(filename, format='pdf')
            filename = "figure/prior_eigenvalue.eps"
            fig.savefig(filename, format='eps')

            plt.close()
