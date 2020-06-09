
from __future__ import absolute_import, division, print_function

import numpy as np
from ..modeling.variables import PARAMETER
from .metric import MetricPrior, MetricPosteriorSeparate, MetricPosteriorAverage
from mpi4py import MPI

# import os
# if not os.path.isdir("data"):
#     os.mkdir("data")
import pickle
import time


class Kernel:
    # evaluate the value and the gradient of the kernel at given particles
    def __init__(self, model, particle, variation, options, comm):

        self.model = model  # forward model
        self.particle = particle  # particle class
        self.variation = variation
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.save_kernel = options["save_kernel"]
        self.delta_kernel = options["delta_kernel"]
        self.delta_kernel_hold = options["delta_kernel"]
        self.max_iter_delta_kernel = options["max_iter_delta_kernel"]
        self.type_Hessian = options["type_Hessian"]
        self.type_metric = options["type_metric"]
        self.type_scaling = options["type_scaling"]
        self.scale = None

        # metric M in k(pn,pm) = exp(-(pn-pm)^T * M * (pn-pm))
        if self.type_metric is 'prior':
            self.metric = MetricPrior(model, particle, variation, options)
        elif self.type_metric is 'posterior_average':
            self.metric = MetricPosteriorAverage(model, particle, variation, options)
        elif self.type_metric is 'posterior_separate':
            self.metric = MetricPosteriorSeparate(model, particle, variation, options)
        else:
            raise NotImplementedError("required metric is not implemented")

        self.value_set = np.empty([self.particle.number_particles_all, self.nproc, self.particle.number_particles_all],
                                  dtype=float)
        if self.save_kernel:
            self.gradient_set = [
                [[self.particle.generate_vector() for m in range(particle.number_particles_all)] for p in
                 range(self.nproc)] for n in range(particle.number_particles_all)]

        self.value_sum = np.empty([self.nproc, self.particle.number_particles_all], dtype=float)
        self.value_set_gather = None
        self.value_sum_reduce = None
        self.gradient_sum = None
        self.gradient_set_gather = None
        self.gradient_sum_reduce = None

        self.time_communication = 0.
        self.time_computation = 0.

    def communication(self):
        time_communication = time.time()

        if self.options["type_optimization"] is not "gradientDescent":
            value_set_array = np.empty([self.particle.number_particles_all, self.nproc, self.particle.number_particles_all], dtype=float)
            value_set_gather_array = np.empty([self.nproc, self.particle.number_particles_all, self.nproc, self.particle.number_particles_all], dtype=float)
            for n in range(self.particle.number_particles_all):
                for p in range(self.nproc):
                    value_set_array[n, p, :] = self.value_set[n][p]
            self.comm.Allgather(value_set_array, value_set_gather_array)  # p n p n

            self.value_set_gather = value_set_gather_array

            if self.options["is_projection"] and self.options["type_Hessian"] is "full":
                gradient_set_array = np.empty(
                    [self.particle.number_particles_all, self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
                gradient_set_gather_array = np.empty(
                    [self.nproc, self.particle.number_particles_all, self.nproc, self.particle.number_particles_all, self.particle.dimension],
                    dtype=float)
                for n in range(self.particle.number_particles_all):
                    for p in range(self.nproc):
                        for m in range(self.particle.number_particles_all):
                            gradient_set_array[n, p, m, :] = self.gradient_set[n][p][m].get_local()
                self.comm.Allgather(gradient_set_array, gradient_set_gather_array)  # p n p n

                self.gradient_set_gather = gradient_set_gather_array

            if self.save_kernel:
                value_sum_array = np.empty([self.nproc, self.particle.number_particles_all], dtype=float)
                value_sum_reduce_array = np.empty([self.nproc, self.particle.number_particles_all], dtype=float)
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        value_sum_array[p, n] = self.value_sum[p][n]
                self.comm.Allreduce(value_sum_array, value_sum_reduce_array, op=MPI.SUM)
                self.value_sum_reduce = value_sum_reduce_array

                gradient_sum_array = np.empty([self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
                gradient_sum_reduce_array = np.empty([self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        gradient_sum_array[p, n, :] = self.gradient_sum[p][n].get_local()
                self.comm.Allreduce(gradient_sum_array, gradient_sum_reduce_array, op=MPI.SUM)

                self.gradient_sum_reduce = [[self.particle.generate_vector() for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles_all):
                        self.gradient_sum_reduce[p][n].set_local(gradient_sum_reduce_array[p, n, :])

        self.time_communication += time.time() - time_communication

    def update(self, particle, variation):

        self.particle = particle
        self.variation = variation

        time_computation = time.time()

        t0 = time.time()
        # update the size of the data if new particles are added
        self.value_set = np.empty([self.particle.number_particles_all, self.nproc, self.particle.number_particles_all],
                                  dtype=float)
        if self.save_kernel:
            self.gradient_set = [
                [[self.particle.generate_vector() for m in range(particle.number_particles_all)] for p in
                 range(self.nproc)] for n in range(particle.number_particles_all)]

        # evaluate M * (pn-pm) and (pn-pm)^T * M * (pn-pm)
        self.scale = np.zeros(particle.number_particles_all)
        grad_norm_set = np.empty([self.particle.number_particles_all, self.nproc, self.particle.number_particles_all], dtype=float)
        phelp = self.particle.generate_vector()
        gradient_set = np.zeros((particle.number_particles_all, self.nproc, particle.number_particles_all, particle.dimension))
        for n in range(particle.number_particles_all):
            if self.options["is_projection"]:
                pn = self.particle.coefficients[n]
                for p in range(self.nproc):
                    for m in range(particle.number_particles_all):
                        pm = particle.coefficients_gather[p][m]
                        phelp.zero()
                        phelp.axpy(1.0, pn)
                        phelp.axpy(-1.0, pm)
                        pout = self.particle.generate_vector()
                        self.metric.mult(phelp, pout, n)
                        self.value_set[n, p, m] = phelp.inner(pout)  # (pn-pm)^T * M * (pn-pm)
                        grad_norm_set[n, p, m] = np.sqrt(pout.inner(pout))
                        if self.save_kernel:
                            self.gradient_set[n][p][m].zero()
                            self.gradient_set[n][p][m].axpy(1., pout)  # M * (pn-pm)
            else:
                if self.options["kernel_vectorized"]:
                    pn = particle.particles_array[n]
                    pm = particle.particles_gather_array
                    pdiff = np.subtract(pn, pm)
                    self.value_set[n, :, :] = np.sum(np.multiply(pdiff, pdiff), axis=2)  # todo include metric matrix !!
                    grad_norm_set[n, :, :] = self.value_set[n, :, :]

                    if self.save_kernel:
                        gradient_set[n] = pdiff
                else:
                    pn = self.particle.particles[n]
                    for p in range(self.nproc):
                        for m in range(particle.number_particles_all):
                            pm = self.particle.particles_gather[p][m]
                            phelp.zero()
                            phelp.axpy(1.0, pn)
                            phelp.axpy(-1.0, pm)
                            pout = self.model.generate_vector(PARAMETER)
                            self.metric.mult(phelp, pout, n)
                            self.value_set[n, p, m] = phelp.inner(pout)  # (pn-pm)^T * M * (pn-pm)
                            phelp.zero()
                            self.model.prior.Msolver.solve(phelp, pout)
                            grad_norm_set[n, p, m] = np.sqrt(pout.inner(phelp))
                            if self.save_kernel:
                                self.gradient_set[n][p][m].zero()
                                self.gradient_set[n][p][m].axpy(1., pout)  # M * (pn-pm)

            if self.type_scaling == 1:
                # rescaling by the parameter dimension = trace(I)
                self.scale[n] = self.particle.dimension
            elif self.type_scaling == 2:
                # rescaling by the trace = trace(I) + trace(H)
                self.scale[n] = (self.particle.dimension + np.sum(self.variation.d[n]))
            elif self.type_scaling == 3:
                # rescaling by the median of the distances (pn-p)^T*M*(pn-p)
                self.scale[n] = np.median(self.value_set[n]) / np.maximum(1, np.log(self.nproc*particle.number_particles))
            elif self.type_scaling == 4:
                # rescaling by the balanced mean mean(M*(pn-p))/mean(gradient of log-posterior)
                self.scale[n] = np.mean(grad_norm_set[n])/np.mean(self.variation.gradient_norm_gather)
                # print("self.scale[n]", self.scale[n], "np.mean(grad_norm_set[n])", np.mean(grad_norm_set[n]),
                # "np.mean(self.variation.gradient_norm_gather)", np.mean(self.variation.gradient_norm_gather))

            if self.scale[n] < 1e-15:
                self.scale[n] = 1.

        self.metric.update(particle, variation)

        # for n in range(particle.number_particles_all):
        #     for p in range(self.nproc):
        #         for m in range(particle.number_particles_all):
        #             # evaluate the kernel k(pn,pm) = exp(-(pn-pm)^T * M * (pn-pm))
        #             if ((p != self.rank) or (m != n)) and self.delta_kernel:
        #                 self.value_set[n, p, m] = 0.
        #             else:
        #                 pass
        #                 self.value_set[n, p, m] = np.exp(-self.value_set[n, p, m]/(2*self.scale[n]))
        #             if self.save_kernel:
        #                 # evaluate the gradient of kernel at (pn, pm) with respect to pm,
        #                 # grad k(pn, pm) = 2 * k(pn, pm) * M * (pn - pm), where M is rescaled by dividing scale
        #                 self.gradient_set[n][p][m][:] *= self.value_set[n][p][m]/self.scale[n]

        for n in range(particle.number_particles_all):
            self.value_set[n, :, :] = np.exp(-self.value_set[n, :, :] / (2 * self.scale[n]))
            if self.save_kernel:
                if self.options["kernel_vectorized"]:
                    gradient_set[n] = np.multiply(gradient_set[n].reshape((self.nproc*particle.number_particles_all,particle.dimension)).T,
                                                  self.value_set[n].reshape(self.nproc*particle.number_particles_all)/self.scale[n]).T.\
                        reshape((self.nproc, particle.number_particles_all, particle.dimension))
                    for p in range(self.nproc):
                        for m in range(particle.number_particles_all):
                            self.gradient_set[n][p][m].set_local(gradient_set[n][p][m])
                else:
                    for p in range(self.nproc):
                        for m in range(particle.number_particles_all):
                            # # evaluate the kernel k(pn,pm) = exp(-(pn-pm)^T * M * (pn-pm))
                            # if ((p != self.rank) or (m != n)) and self.delta_kernel:
                            #     self.value_set[n, p, m] = 0.
                            # else:
                            #     # pass
                            #     self.value_set[n, p, m] = np.exp(-self.value_set[n, p, m]/(2*self.scale[n]))
                            # # evaluate the gradient of kernel at (pn, pm) with respect to pm,
                            # # grad k(pn, pm) = 2 * k(pn, pm) * M * (pn - pm), where M is rescaled by dividing scale
                            if self.save_kernel:
                                self.gradient_set[n][p][m][:] *= self.value_set[n][p][m]/self.scale[n]

            if self.options["print_level"] > 10:
                print("scale = ", self.scale[n], " kernel = ", [j for sub in self.value_set[n] for j in sub])

        # save the lumped kernel and gradient
        if self.save_kernel:
            self.value_sum = np.zeros((self.nproc, self.particle.number_particles_all))
            self.gradient_sum = [
                [self.particle.generate_vector() for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
            for p in range(self.nproc):
                for m in range(particle.number_particles_all):
                    self.value_sum[p, m] = 0.
                    for n in range(particle.number_particles):
                        self.value_sum[p, m] = self.value_sum[p, m] + self.value_set[n, p, m]
                        self.gradient_sum[p][m].axpy(1.0, self.gradient_set[n][p][m])

        t1 = time.time()-t0
        if self.options["print_level"] > 2:
            print("time to evaluate " + str(self.particle.number_particles_all) + " kernel and gradients = ", t1)

        self.time_computation += time.time() - time_computation

        t0 = time.time()
        self.communication()
        t1 = time.time()-t0
        if self.options["print_level"] > 2:
            print("time to perform communication for kernel data in " + str(self.nproc) + " processors = ", t1)

    def value(self, n=0, p=0, m=0):
        # evaluate the kernel k(pn,pm) = exp(-(pn-pm)^T * M * (pn-pm))
        if self.options["is_projection"]:
            pn = self.particle.coefficients[n]  # n = 1, ..., particle.number_particles
            pm = self.particle.coefficients_gather[p][m]  # m = 1, ..., particle.number_particles_all
        else:
            pn = self.particle.particles[n]  # n = 1, ..., particle.number_particles
            pm = self.particle.particles_gather[p][m]  # m = 1, ..., particle.number_particles_all
        phelp = self.particle.generate_vector()
        phelp.zero()
        phelp.axpy(1.0, pn)
        phelp.axpy(-1.0, pm)

        value = np.exp(-self.metric.inner(phelp, phelp, n=n)/(2*self.scale[n]))  # k(pn,pm) = exp(-(pn-pm)^T * M * (pn-pm))

        return value

    def values(self, n=0):
        # evaluate the kernel at (pn, pm) for given n and all m = 1, ..., N
        values = [[None for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
        for p in range(self.nproc):
            for m in range(self.particle.number_particles_all):
                if (p != self.rank or m != n) and self.delta_kernel:  # set zero of small kernels
                    values[p][m] = 0
                else:
                    values[p][m] = self.value(n=n, p=p, m=m)
        # print("kernel values = ", values)
        self.value_set[n] = values

        return values

    def gradient(self, n=0, p=0, m=0):
        # evaluate the gradient of kernel at (pn, pm) with respect to pm, grad k(pn, pm) = 2 * k(pn, pm) * M * (pn - pm)
        if self.options["is_projection"]:
            pn = self.particle.coefficients[n]  # n = 1, ..., particle.number_particles
            pm = self.particle.coefficients_gather[p][m]  # m = 1, ..., particle.number_particles_all
        else:
            pn = self.particle.particles[n]  # n = 1, ..., particle.number_particles
            pm = self.particle.particles_gather[p][m]  # m = 1, ..., particle.number_particles_all

        ghelp = self.particle.generate_vector()
        gradient = self.particle.generate_vector()
        phelp = self.particle.generate_vector()
        phelp.zero()
        phelp.axpy(1.0, pn)
        phelp.axpy(-1.0, pm)

        self.metric.mult(phelp, ghelp, n=n)
        value = self.value(n=n, p=p, m=m)
        gradient.axpy(value/self.scale[n], ghelp)

        return gradient

    def gradients(self, n=0):
        # evaluate the gradients of kernel at (pn, pm) with respect to pm, for all m = 1, ..., N
        gradients = [[None for n in range(self.particle.number_particles_all)] for p in range(self.nproc)]
        for p in range(self.nproc):
            for m in range(self.particle.number_particles_all):
                gradients[p][m] = self.particle.generate_vector()
                if (p != self.rank or m != n) and self.delta_kernel:
                    gradients[p][m].zero()
                else:
                    gradients[p][m].axpy(1.0, self.gradient(n=n, p=p, m=m))

        if self.save_kernel:
            self.gradient_set[n] = gradients

        return gradients

    def save_values(self, save_number, it):

        if False:
            for n in range(save_number):
                filename = 'data/rank_' + str(self.rank) + '_kernel_' + str(n) + '_iteration_' + str(it) + '.p'
                pickle.dump(self.value_set, open(filename, 'wb'))

    def hessian(self):
        raise NotImplementedError(" the hessian needs to be implemented ")
