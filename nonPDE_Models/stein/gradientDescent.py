
from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
from .variation import LowRankOperator
from mpi4py import MPI
import time


class GradientDescent:
    # solve the optimization problem by Newton method with separated linear system
    def __init__(self, model, particle, variation, kernel, options, comm):
        self.model = model  # forward model
        self.particle = particle  # set of particles, pn = particles[m]
        self.variation = variation
        self.kernel = kernel
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.save_kernel = options["save_kernel"]
        self.add_number = options["add_number"]
        self.add_step = options["add_step"]
        self.save_step = options["save_step"]
        self.save_number = options["save_number"]
        self.search_size = options["search_size"]
        self.plot = options["plot"]

        self.it = 0
        self.converged = False
        self.reason = 0

        self.gradient_norm = np.zeros(self.particle.number_particles_all)
        self.gradient_norm_init = np.zeros(self.particle.number_particles_all)
        self.pg_phat = np.zeros(self.particle.number_particles_all)
        self.pstep_norm = np.zeros(self.particle.number_particles_all)
        self.step_norm = np.zeros(self.particle.number_particles_all)
        self.step_norm_init = np.zeros(self.particle.number_particles_all)
        self.relative_grad_norm = np.zeros(self.particle.number_particles_all)
        self.relative_step_norm = np.zeros(self.particle.number_particles_all)
        self.tol_gradient = np.zeros(self.particle.number_particles_all)
        self.final_grad_norm = np.zeros(self.particle.number_particles_all)
        self.cost_new = np.zeros(self.particle.number_particles_all)
        self.reg_new = np.zeros(self.particle.number_particles_all)
        self.misfit_new = np.zeros(self.particle.number_particles_all)
        self.alpha = 1.e-2 * np.ones(self.particle.number_particles_all)
        self.n_backtrack = np.zeros(self.particle.number_particles_all)

        self.data_save = dict()

        self.data_save["nCores"] = self.nproc
        self.data_save["particle_dimension"] = particle.particle_dimension
        self.data_save["dimension"] = []
        self.data_save["gradient_norm"] = []
        self.data_save["pg_phat"] = []
        self.data_save["step_norm"] = []
        self.data_save["relative_grad_norm"] = []
        self.data_save["relative_step_norm"] = []
        self.data_save["cost_new"] = []
        self.data_save["reg_new"] = []
        self.data_save["misfit_new"] = []
        self.data_save["iteration"] = []
        self.data_save["meanL2norm"] = []
        self.data_save["moment2L2norm"] = []
        self.data_save["meanErrorL2norm"] = []
        self.data_save["varianceL2norm"] = []
        self.data_save["varianceErrorL2norm"] = []
        self.data_save["sample_trace"] = []
        self.data_save["cost_mean"] = []
        self.data_save["cost_std"] = []
        self.data_save["cost_moment2"] = []
        self.data_save["d"] = []
        self.data_save["d_average"] = []

        self.filename = "data/data_nSamples_" + str(self.nproc*self.particle.number_particles_all) \
                        + "_isProjection_" + str(self.options["is_projection"]) + "_SVGD.p"

        if self.model.qoi is not None:
            self.data_save["qoi_std"] = []
            self.data_save["qoi_mean"] = []
            self.data_save["qoi_moment2"] = []
            self.qoi_mean = 0.
            self.qoi_std = 0.
            self.qoi_moment2 = 0.

        self.time_communication = 0.
        self.time_computation = 0.

    def gradientSeparated(self, gradient, m):

        if self.save_kernel:
            kernel_value = self.kernel.value_set[m]
            kernel_gradient = self.kernel.gradient_set[m]
        else:
            kernel_value = self.kernel.values(m)
            kernel_gradient = self.kernel.gradients(m)

        if self.kernel.delta_kernel:
            gp_misfit = self.particle.generate_vector()
            gp_misfit[:] = self.variation.gradient_gather[self.rank][m]
            gradient += kernel_value[self.rank][m] * gp_misfit
            gradient += -1.0 * kernel_gradient[self.rank][m]
        else:
            for p in range(self.nproc):
                for ie in range(self.particle.number_particles):  # take the expectation over particle set
                    gp_misfit = self.particle.generate_vector()
                    gp_misfit[:] = self.variation.gradient_gather[p][ie]
                    gradient += kernel_value[p][ie] * gp_misfit
                    gradient += -1.0 * kernel_gradient[p][ie]

            # also use the particle to compute the expectation
            if m >= self.particle.number_particles:
                gp_misfit = self.particle.generate_vector()
                gp_misfit[:] = self.variation.gradient_gather[self.rank][m]
                gradient += kernel_value[self.rank][m] * gp_misfit
                gradient += -1.0 * kernel_gradient[self.rank][m]

            # this is not needed in Newton method, because both sides are divided by # particles
            gradient[:] /= self.nproc*self.particle.number_particles + (m >= self.particle.number_particles)

        if self.options["is_projection"]:
            if self.options["is_precondition"]:
                if self.options["type_approximation"] is 'hessian':
                    A = self.variation.hessian_misfit_average + np.eye(self.variation.hessian_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient)
                    gradient[:] = gradient_array
                elif self.options["type_approximation"] is 'fisher':
                    A = self.variation.fisher_misfit_average + np.eye(self.variation.fisher_misfit_average.shape[0])
                    gradient_array = np.linalg.solve(A, gradient)
                    gradient[:] = gradient_array

            gradient_norm = np.sqrt(gradient.dot(gradient))
        else:
            if self.options["is_precondition"]:
                gradient_tmp = self.model.generate_vector()
                gradient_tmp += 1.0 * gradient
                d = np.divide(self.variation.d_average, (1+self.variation.d_average))
                hessian_misfit = LowRankOperator(d, self.variation.U_average)
                gradient_misfit = self.model.generate_vector()
                hessian_misfit.mult(gradient_tmp, gradient_misfit)

                self.model.prior.Rsolver.solve(gradient, gradient_tmp)
                gradient += -1.0 * gradient_misfit

            tmp = self.particle.generate_vector()
            self.model.prior.Msolver.solve(tmp, gradient)
            gradient_norm = np.sqrt(gradient.dot(tmp))

        return gradient_norm

    def communication(self, phat):

        time_communication = time.time()

        phat_array = np.empty([self.particle.number_particles_all, self.particle.dimension], dtype=float)
        phat_gather_array = np.empty([self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
        for n in range(self.particle.number_particles_all):
            phat_array[n, :] = phat[n]
        self.comm.Allgather(phat_array, phat_gather_array)

        phat_gather = [[self.particle.generate_vector() for n in range(self.particle.number_particles_all)] for p
                       in range(self.nproc)]
        for p in range(self.nproc):
            for n in range(self.particle.number_particles_all):
                phat_gather[p][n][:] = phat_gather_array[p, n, :]

        self.time_communication += time.time() - time_communication

        return phat_gather

    def save(self):

        if self.save_number and np.mod(self.it, self.save_step) == 0:
            # self.kernel.save_values(self.save_number, self.it)
            self.particle.save(self.save_number, self.it)
            # self.variation.save_eigenvalue(self.save_number, self.it)
            if self.plot:
                self.variation.plot_eigenvalue(self.save_number, self.it)
            if self.rank == 0:
                pickle.dump(self.data_save, open(self.filename, 'wb'))

        if self.options["type_parameter"] is 'vector' and self.plot:
            self.particle.plot_particles(self.particle, self.it)

    def solve(self):
        # use gradient decent method to solve the optimization problem
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]
        line_search = self.options["line_search"]
        c_armijo = self.options["c_armijo"]
        max_backtracking_iter = self.options["max_backtracking_iter"]

        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)

        self.it = 0
        self.converged = False

        self.save()

        self.cost_mean, self.cost_std = 0., 0.

        while self.it < max_iter and (self.converged is False):

            # 1 data statistics
            time_test = time.time()


            if self.particle.mean_posterior is None:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace = self.particle.statistics()
                self.meanErrorL2norm, self.varianceErrorL2norm = 0., 0.
            else:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace, \
                self.meanErrorL2norm, self.varianceErrorL2norm = self.particle.statistics()

            if self.rank == 0:
                print("# samples = ", self.nproc * self.particle.number_particles_all,
                      " mean = ", self.meanL2norm, "mean error = ", self.meanErrorL2norm,
                      " variance = ", self.varianceL2norm, " variance error = ", self.varianceErrorL2norm,
                      " trace = ", self.sample_trace)

            if self.model.qoi is not None:
                self.qoi_mean, self.qoi_std, self.qoi_moment2 = self.variation.qoi_statistics()

                if self.rank == 0:
                    print("# samples = ", "qoi_mean = ", self.qoi_mean, "qoi_std = ", self.qoi_std,
                          "cost_mean = ", self.cost_mean, "cost_std = ", self.cost_std)

            time_computation = time.time()
            if print_level >= 5:
                print("1. time for data statistics = ", time.time() - time_test)

            # 2. time to compute gradient
            time_test = time.time()

            phat = [self.particle.generate_vector() for m in range(self.particle.number_particles)]
            self.pg_phat = np.ones(self.particle.number_particles)
            for m in range(self.particle.number_particles):  # solve for each particle
                # evaluate gradient
                gradient = self.particle.generate_vector()
                self.gradient_norm[m] = self.gradientSeparated(gradient, m)

                phat[m] += -1.0 * gradient
                self.pg_phat[m] = gradient.dot(phat[m])

                # set tolerance for gradient iteration
                if self.it == 0:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                if self.particle.number_particles > self.particle.number_particles_old:
                        self.gradient_norm_init[m] = self.gradient_norm[m]
                        self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

            phat = self.communication(phat)  # gather the coefficients to all processors
            if print_level >= 5:
                print("2. time to compute gradient = ", time.time() - time_test)

            # 3. time for particle update
            time_test = time.time()

            # step for particle update, pstep(x) = sum_n phat_n k_n(x)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            deltap = [self.model.generate_vector() for m in range(self.particle.number_particles_all)]
            for m in range(self.particle.number_particles_all):
                pstep[m] += 1.0 * phat[self.rank][m]

                # for p in range(self.nproc):
                #     for n in range(self.particle.number_particles):
                #         pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat[p][n])
                #     if m >= self.particle.number_particles:
                #         pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat[p][m])

                if self.options["is_projection"]:
                    self.pstep_norm[m] = np.sqrt(pstep[m].dot(pstep[m]))
                    pstep_m = pstep[m]
                    for r in range(self.particle.coefficient_dimension):
                        deltap[m] += pstep_m[r] * self.particle.bases[r]
                else:
                    phelp = self.model.generate_vector()
                    self.model.prior.M.mult(pstep[m], phelp)
                    self.pstep_norm[m] = np.sqrt(pstep[m].dot(phelp))
                    deltap[m] += 1.0 * pstep[m]

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    if m >= self.particle.number_particles_old \
                            and m < self.particle.number_particles_all - self.particle.number_particles_add:
                        self.step_norm_init[m] = self.pstep_norm[m]

            if self.it == 0:
                self.step_norm_init = self.pstep_norm

            self.alpha = self.search_size * np.ones(self.particle.number_particles_all)
            # self.alpha = np.power(self.it+1., -0.5) * self.search_size * np.ones(self.particle.number_particles_all) #gradient_norm_max_one #

            self.n_backtrack = np.zeros(self.particle.number_particles_all)
            self.cost_new = np.zeros(self.particle.number_particles_all)
            self.reg_new = np.zeros(self.particle.number_particles_all)
            self.misfit_new = np.zeros(self.particle.number_particles_all)
            if print_level >= 5:
                print("3. time for particle update = ", time.time() - time_test)

            # 4. time for line search
            time_test = time.time()

            for m in range(self.particle.number_particles_all):
                # compute the old cost
                x = self.particle.particles[m]
                cost_old, reg_old, misfit_old = self.model.cost(x)
                self.cost_new[m], self.reg_new[m], self.misfit_new[m] = cost_old, reg_old, misfit_old

                if line_search:
                    # do line search
                    descent = 0
                    x_star = self.model.generate_vector()
                    while descent == 0 and self.n_backtrack[m] < max_backtracking_iter:
                        # update the parameter
                        x_star[:] = 0.
                        x_star += x
                        x_star += self.alpha[m] * deltap[m]
                        # update the state at new parameter

                        # evaluate the cost functional, here the potential
                        self.cost_new[m], self.reg_new[m], self.misfit_new[m] = self.model.cost(x_star)

                        # Check if armijo conditions are satisfied
                        if m < self.particle.number_particles:
                            if (self.cost_new[m] < cost_old + self.alpha[m] * c_armijo * self.pg_phat[m]) or \
                                    (-self.pg_phat[m] <= self.options["gdm_tolerance"]):
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5
                                # print("alpha = ", alpha[m])
                        else:  # we do not have pg_phat for m >= particle.number_particles
                            if self.cost_new[m] < cost_old:
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5

            self.cost_mean, self.cost_std, self.cost_moment2 = np.mean(self.cost_new), np.std(self.cost_new), np.mean(self.cost_new**2)
            if print_level >= 5:
                print("4. time for line search = ", time.time() - time_test)

            # 5. save data
            time_test = time.time()


            # compute the norm of the step/direction to move
            self.step_norm = self.pstep_norm * self.alpha

            # move all particles in the new directions, pm = pm + self.alpha[m] * sum_n phat[n] * k(pn, pm)
            self.particle.move(self.alpha, pstep)

            self.relative_grad_norm = np.divide(self.gradient_norm, self.gradient_norm_init)
            self.relative_step_norm = np.divide(self.step_norm, self.step_norm_init)

            self.time_computation += time.time() - time_computation

            # print data
            if print_level >= -1:
                if self.rank == 0:
                    print("\n{0:5} {1:5} {2:8} {3:15} {4:15} {5:15} {6:15} {7:15} {8:14}".format(
                        "it", "cpu", "id", "cost", "misfit", "reg", "||g||L2", "||m||L2", "alpha"))
                for m in range(self.particle.number_particles):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e} {8:14e}".format(
                    self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.relative_grad_norm[m], self.relative_step_norm[m], self.alpha[m]))
                for m in range(self.particle.number_particles, self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m], 0., self.alpha[m]))

            # save data
            gradient_norm = np.empty([self.nproc, len(self.gradient_norm)], dtype=float)
            self.comm.Allgather(self.gradient_norm, gradient_norm)
            pg_phat = np.empty([self.nproc, len(self.pg_phat)], dtype=float)
            self.comm.Allgather(self.pg_phat, pg_phat)
            step_norm = np.empty([self.nproc, len(self.step_norm)], dtype=float)
            self.comm.Allgather(self.step_norm, step_norm)
            relative_grad_norm = np.empty([self.nproc, len(self.relative_grad_norm)], dtype=float)
            self.comm.Allgather(self.relative_grad_norm, relative_grad_norm)
            relative_step_norm = np.empty([self.nproc, len(self.relative_step_norm)], dtype=float)
            self.comm.Allgather(self.relative_step_norm, relative_step_norm)
            cost_new = np.empty([self.nproc, len(self.cost_new)], dtype=float)
            self.comm.Allgather(self.cost_new, cost_new)

            if self.rank == 0:
                self.data_save["gradient_norm"].append(np.mean(gradient_norm))
                # self.data_save["pg_phat"].append(pg_phat)
                self.data_save["step_norm"].append(np.mean(step_norm))
                # self.data_save["relative_grad_norm"].append(relative_grad_norm)
                # self.data_save["relative_step_norm"].append(relative_step_norm)
                # self.data_save["cost_new"].append(cost_new)
                self.data_save["dimension"].append(self.particle.dimension)
                self.data_save["cost_new"].append(cost_new)
                self.data_save["iteration"].append(self.it)
                self.data_save["meanL2norm"].append(self.meanL2norm)
                self.data_save["moment2L2norm"].append(self.moment2L2norm)
                self.data_save["meanErrorL2norm"].append(self.meanErrorL2norm)
                self.data_save["varianceL2norm"].append(self.varianceL2norm)
                self.data_save["varianceErrorL2norm"].append(self.varianceErrorL2norm)
                self.data_save["sample_trace"].append(self.sample_trace)
                self.data_save["cost_mean"].append(self.cost_mean)
                self.data_save["cost_std"].append(self.cost_std)
                self.data_save["cost_moment2"].append(self.cost_moment2)
                self.data_save["d"].append(self.variation.d)
                self.data_save["d_average"].append(self.variation.d_average_save)
                if self.model.qoi is not None:
                    self.data_save["qoi_mean"].append(self.qoi_mean)
                    self.data_save["qoi_std"].append(self.qoi_std)
                    self.data_save["qoi_moment2"].append(self.qoi_moment2)

                N = self.nproc*self.particle.number_particles_all

            # verify stopping criteria
            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if self.gradient_norm[m] > self.tol_gradient[m]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles_all):  # should use _all
                if self.n_backtrack[m] < max_backtracking_iter:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = False
                self.reason = 2
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if -self.pg_phat[m] > self.options["gdm_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 3
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                if self.step_norm[m] > self.step_norm_init[m]*self.options["step_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 4
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            # update data for optimization in next step
            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            if print_level >= 5:
                print("5. time to save data = ", time.time() - time_test)

            # 6. update variation, kernel, particle
            time_test = time.time()

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.gradient_norm = np.insert(self.gradient_norm, m-self.particle.number_particles_add, 0.)
                        self.gradient_norm_init = np.insert(self.gradient_norm_init, m-self.particle.number_particles_add, 0.)
                        self.step_norm_init = np.insert(self.step_norm_init, m-self.particle.number_particles_add, 0.)
                        self.final_grad_norm = np.insert(self.final_grad_norm, m-self.particle.number_particles_add, 0.)

            # update variation, kernel, and hessian before solving the Newton linear system
            self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
            relative_step_norm = np.max(self.relative_step_norm)
            relative_step_norm_reduce = np.zeros(1, dtype=float)
            self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
            self.relative_step_norm = relative_step_norm_reduce[0]
            self.variation.update(self.particle, self.it, self.relative_step_norm)
            self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
            self.kernel.update(self.particle, self.variation)

            # save the particles for visualization and plot the eigenvalues at each particle
            self.save()

            if print_level >= 5:
                print("6. time to update variation, kernel = ", time.time() - time_test)

    def solve_rmsprop(self, step_size=0.1, gamma=0.9, eps=1e-8):
        # use gradient decent method to solve the optimization problem
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]
        line_search = self.options["line_search"]
        c_armijo = self.options["c_armijo"]
        max_backtracking_iter = self.options["max_backtracking_iter"]

        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)

        self.it = 0
        self.converged = False

        self.save()

        self.cost_mean, self.cost_std = 0., 0.

        avg_sq_grad = np.ones((self.particle.number_particles_all, self.particle.dimension))
        while self.it < max_iter and (self.converged is False):

            if self.particle.mean_posterior is None:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace = self.particle.statistics()
                self.meanErrorL2norm, self.varianceErrorL2norm = 0., 0.
            else:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace, \
                self.meanErrorL2norm, self.varianceErrorL2norm = self.particle.statistics()

            if self.rank == 0:
                print("# samples = ", self.nproc * self.particle.number_particles_all,
                      " mean = ", self.meanL2norm, "mean error = ", self.meanErrorL2norm,
                      " variance = ", self.varianceL2norm, " variance error = ", self.varianceErrorL2norm,
                      " trace = ", self.sample_trace)

            if self.model.qoi is not None:
                self.qoi_mean, self.qoi_std, self.qoi_moment2 = self.variation.qoi_statistics()

                if self.rank == 0:
                    print("# samples = ", "qoi_mean = ", self.qoi_mean, "qoi_std = ", self.qoi_std,
                          "cost_mean = ", self.cost_mean, "cost_std = ", self.cost_std)

            time_computation = time.time()

            phat = [self.particle.generate_vector() for m in range(self.particle.number_particles)]
            self.pg_phat = np.ones(self.particle.number_particles)
            for m in range(self.particle.number_particles):  # solve for each particle
                # evaluate gradient
                gradient = self.particle.generate_vector()
                self.gradient_norm[m] = self.gradientSeparated(gradient, m)

                avg_sq_grad[m, :] = avg_sq_grad[m, :] * gamma + gradient ** 2 * (1 - gamma)

                phat[m] += -1.0 * gradient
                self.pg_phat[m] = gradient.dot(phat[m])

                # set tolerance for gradient iteration
                if self.it == 0:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                if self.particle.number_particles > self.particle.number_particles_old:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

            phat = self.communication(phat)  # gather the coefficients to all processors

            # step for particle update, pstep(x) = sum_n phat_n k_n(x)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            deltap = [self.model.generate_vector() for m in range(self.particle.number_particles_all)]
            for m in range(self.particle.number_particles_all):
                pstep[m] += 1.0 * np.divide(phat[self.rank][m], np.sqrt(avg_sq_grad[m, :]) + eps)

                # for p in range(self.nproc):
                #     for n in range(self.particle.number_particles):
                #         pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat[p][n])
                #     if m >= self.particle.number_particles:
                #         pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat[p][m])

                if self.options["is_projection"]:
                    self.pstep_norm[m] = np.sqrt(pstep[m].dot(pstep[m]))
                    pstep_m = pstep[m]
                    for r in range(self.particle.coefficient_dimension):
                        deltap[m] += pstep_m[r] * self.particle.bases[r]
                else:
                    phelp = self.model.generate_vector()
                    self.model.prior.M.mult(pstep[m], phelp)
                    self.pstep_norm[m] = np.sqrt(pstep[m].dot(phelp))
                    deltap[m] += 1.0 * pstep[m]

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    if m >= self.particle.number_particles_old \
                            and m < self.particle.number_particles_all - self.particle.number_particles_add:
                        self.step_norm_init[m] = self.pstep_norm[m]

            if self.it == 0:
                self.step_norm_init = self.pstep_norm

            self.alpha = step_size * np.ones(self.particle.number_particles_all)
            # self.alpha = np.power(self.it+1., -0.5) * self.search_size * np.ones(self.particle.number_particles_all) #gradient_norm_max_one #

            self.n_backtrack = np.zeros(self.particle.number_particles_all)
            self.cost_new = np.zeros(self.particle.number_particles_all)
            self.reg_new = np.zeros(self.particle.number_particles_all)
            self.misfit_new = np.zeros(self.particle.number_particles_all)

            for m in range(self.particle.number_particles_all):
                # compute the old cost
                x = self.particle.particles[m]
                cost_old, reg_old, misfit_old = self.model.cost(x)
                self.cost_new[m], self.reg_new[m], self.misfit_new[m] = cost_old, reg_old, misfit_old

                line_search = False
                if line_search:
                    # do line search
                    descent = 0
                    x_star = self.model.generate_vector()
                    while descent == 0 and self.n_backtrack[m] < max_backtracking_iter:
                        # update the parameter
                        x_star[:] = 0.
                        x_star += x
                        x_star += self.alpha[m] * deltap[m]
                        # update the state at new parameter

                        # evaluate the cost functional, here the potential
                        self.cost_new[m], self.reg_new[m], self.misfit_new[m] = self.model.cost(x_star)

                        # Check if armijo conditions are satisfied
                        if m < self.particle.number_particles:
                            if (self.cost_new[m] < cost_old + self.alpha[m] * c_armijo * self.pg_phat[m]) or \
                                    (-self.pg_phat[m] <= self.options["gdm_tolerance"]):
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5
                                # print("alpha = ", alpha[m])
                        else:  # we do not have pg_phat for m >= particle.number_particles
                            if self.cost_new[m] < cost_old:
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5

            self.cost_mean, self.cost_std, self.cost_moment2 = np.mean(self.cost_new), np.std(self.cost_new), np.mean(
                self.cost_new ** 2)

            # compute the norm of the step/direction to move
            self.step_norm = self.pstep_norm * self.alpha

            # move all particles in the new directions, pm = pm + self.alpha[m] * sum_n phat[n] * k(pn, pm)
            self.particle.move(self.alpha, pstep)

            self.relative_grad_norm = np.divide(self.gradient_norm, self.gradient_norm_init)
            self.relative_step_norm = np.divide(self.step_norm, self.step_norm_init)

            self.time_computation += time.time() - time_computation

            # print data
            if print_level >= -1:
                if self.rank == 0:
                    print("\n{0:5} {1:5} {2:8} {3:15} {4:15} {5:15} {6:15} {7:15} {8:14}".format(
                        "it", "cpu", "id", "cost", "misfit", "reg", "||g||L2", "||m||L2", "alpha"))
                for m in range(self.particle.number_particles):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e} {8:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.relative_grad_norm[m], self.relative_step_norm[m], self.alpha[m]))
                for m in range(self.particle.number_particles, self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m], 0.,
                        self.alpha[m]))

            # save data
            gradient_norm = np.empty([self.nproc, len(self.gradient_norm)], dtype=float)
            self.comm.Allgather(self.gradient_norm, gradient_norm)
            pg_phat = np.empty([self.nproc, len(self.pg_phat)], dtype=float)
            self.comm.Allgather(self.pg_phat, pg_phat)
            step_norm = np.empty([self.nproc, len(self.step_norm)], dtype=float)
            self.comm.Allgather(self.step_norm, step_norm)
            relative_grad_norm = np.empty([self.nproc, len(self.relative_grad_norm)], dtype=float)
            self.comm.Allgather(self.relative_grad_norm, relative_grad_norm)
            relative_step_norm = np.empty([self.nproc, len(self.relative_step_norm)], dtype=float)
            self.comm.Allgather(self.relative_step_norm, relative_step_norm)
            cost_new = np.empty([self.nproc, len(self.cost_new)], dtype=float)
            self.comm.Allgather(self.cost_new, cost_new)

            if self.rank == 0:
                self.data_save["gradient_norm"].append(np.mean(gradient_norm))
                # self.data_save["pg_phat"].append(pg_phat)
                self.data_save["step_norm"].append(np.mean(step_norm))
                # self.data_save["relative_grad_norm"].append(relative_grad_norm)
                # self.data_save["relative_step_norm"].append(relative_step_norm)
                # self.data_save["cost_new"].append(cost_new)
                self.data_save["dimension"].append(self.particle.dimension)
                self.data_save["cost_new"].append(cost_new)
                self.data_save["iteration"].append(self.it)
                self.data_save["meanL2norm"].append(self.meanL2norm)
                self.data_save["moment2L2norm"].append(self.moment2L2norm)
                self.data_save["meanErrorL2norm"].append(self.meanErrorL2norm)
                self.data_save["varianceL2norm"].append(self.varianceL2norm)
                self.data_save["varianceErrorL2norm"].append(self.varianceErrorL2norm)
                self.data_save["sample_trace"].append(self.sample_trace)
                self.data_save["cost_mean"].append(self.cost_mean)
                self.data_save["cost_std"].append(self.cost_std)
                self.data_save["cost_moment2"].append(self.cost_moment2)
                self.data_save["d"].append(self.variation.d)
                self.data_save["d_average"].append(self.variation.d_average_save)
                if self.model.qoi is not None:
                    self.data_save["qoi_mean"].append(self.qoi_mean)
                    self.data_save["qoi_std"].append(self.qoi_std)
                    self.data_save["qoi_moment2"].append(self.qoi_moment2)

                # N = self.nproc * self.particle.number_particles_all
                # filename = "data/data_nSamples_" + str(N) + "_isProjection_" + str(
                #     self.options["is_projection"]) + "_SVGD.p"

            # verify stopping criteria
            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if self.gradient_norm[m] > self.tol_gradient[m]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles_all):  # should use _all
                if self.n_backtrack[m] < max_backtracking_iter:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = False
                self.reason = 2
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if -self.pg_phat[m] > self.options["gdm_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 3
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                if self.step_norm[m] > self.step_norm_init[m] * self.options["step_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 4
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            # update data for optimization in next step
            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.gradient_norm = np.insert(self.gradient_norm, m - self.particle.number_particles_add, 0.)
                        self.gradient_norm_init = np.insert(self.gradient_norm_init,
                                                            m - self.particle.number_particles_add, 0.)
                        self.step_norm_init = np.insert(self.step_norm_init, m - self.particle.number_particles_add, 0.)
                        self.final_grad_norm = np.insert(self.final_grad_norm, m - self.particle.number_particles_add,
                                                         0.)

            # update variation, kernel, and hessian before solving the Newton linear system
            self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
            relative_step_norm = np.max(self.relative_step_norm)
            relative_step_norm_reduce = np.zeros(1, dtype=float)
            self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
            self.relative_step_norm = relative_step_norm_reduce[0]
            self.variation.update(self.particle, self.it, self.relative_step_norm)
            self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
            self.kernel.update(self.particle, self.variation)

            # save the particles for visualization and plot the eigenvalues at each particle
            self.save()

    def solve_adam(self, step_size=0.01, b1=0.9, b2=0.999, eps=1e-8):
        # use gradient decent method to solve the optimization problem
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]
        line_search = self.options["line_search"]
        c_armijo = self.options["c_armijo"]
        max_backtracking_iter = self.options["max_backtracking_iter"]

        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)

        self.it = 0
        self.converged = False

        self.save()

        self.cost_mean, self.cost_std = 0., 0.

        m_adam = np.ones((self.particle.number_particles_all, self.particle.dimension))
        v_adam = np.ones((self.particle.number_particles_all, self.particle.dimension))
        m_hat = np.ones((self.particle.number_particles_all, self.particle.dimension))
        v_hat = np.ones((self.particle.number_particles_all, self.particle.dimension))
        while self.it < max_iter and (self.converged is False):

            if self.particle.mean_posterior is None:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace = self.particle.statistics()
                self.meanErrorL2norm, self.varianceErrorL2norm = 0., 0.
            else:
                mean, self.meanL2norm, self.moment2L2norm, variance, self.varianceL2norm, self.sample_trace, \
                self.meanErrorL2norm, self.varianceErrorL2norm = self.particle.statistics()

            if self.rank == 0:
                print("# samples = ", self.nproc * self.particle.number_particles_all,
                      " mean = ", self.meanL2norm, "mean error = ", self.meanErrorL2norm,
                      " variance = ", self.varianceL2norm, " variance error = ", self.varianceErrorL2norm,
                      " trace = ", self.sample_trace)

            if self.model.qoi is not None:
                self.qoi_mean, self.qoi_std, self.qoi_moment2 = self.variation.qoi_statistics()

                if self.rank == 0:
                    print("# samples = ", "qoi_mean = ", self.qoi_mean, "qoi_std = ", self.qoi_std,
                          "cost_mean = ", self.cost_mean, "cost_std = ", self.cost_std)

            time_computation = time.time()

            phat = [self.particle.generate_vector() for m in range(self.particle.number_particles)]
            self.pg_phat = np.ones(self.particle.number_particles)
            for m in range(self.particle.number_particles):  # solve for each particle
                # evaluate gradient
                gradient = self.particle.generate_vector()
                self.gradient_norm[m] = self.gradientSeparated(gradient, m)
                m_adam[m, :] = (1 - b1) * gradient + b1 * m_adam[m, :]
                v_adam[m, :] = (1 - b2) * gradient ** 2 + b2 * v_adam[m, :]
                m_hat[m, :] = m_adam[m, :] / (1 - b1 ** (self.it + 1))
                v_hat[m, :] = v_adam[m, :] / (1 - b2 ** (self.it + 1))

                phat[m] += -1.0 * gradient
                self.pg_phat[m] = gradient.dot(phat[m])

                # set tolerance for gradient iteration
                if self.it == 0:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                if self.particle.number_particles > self.particle.number_particles_old:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_gradient[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

            phat = self.communication(phat)  # gather the coefficients to all processors

            # step for particle update, pstep(x) = sum_n phat_n k_n(x)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            deltap = [self.model.generate_vector() for m in range(self.particle.number_particles_all)]
            for m in range(self.particle.number_particles_all):
                pstep[m] += - np.divide(m_hat[m, :], (np.sqrt(v_hat[m, :]) + eps))

                # for p in range(self.nproc):
                #     for n in range(self.particle.number_particles):
                #         pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat[p][n])
                #     if m >= self.particle.number_particles:
                #         pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat[p][m])

                if self.options["is_projection"]:
                    self.pstep_norm[m] = np.sqrt(pstep[m].dot(pstep[m]))
                    pstep_m = pstep[m]
                    for r in range(self.particle.coefficient_dimension):
                        deltap[m] += pstep_m[r] * self.particle.bases[r]
                else:
                    phelp = self.model.generate_vector()
                    self.model.prior.M.mult(pstep[m], phelp)
                    self.pstep_norm[m] = np.sqrt(pstep[m].dot(phelp))
                    deltap[m] += 1.0 * pstep[m]

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    if m >= self.particle.number_particles_old \
                            and m < self.particle.number_particles_all - self.particle.number_particles_add:
                        self.step_norm_init[m] = self.pstep_norm[m]

            if self.it == 0:
                self.step_norm_init = self.pstep_norm

            self.alpha = step_size * np.ones(self.particle.number_particles_all)
            # self.alpha = np.power(self.it+1., -0.5) * self.search_size * np.ones(self.particle.number_particles_all) #gradient_norm_max_one #

            self.n_backtrack = np.zeros(self.particle.number_particles_all)
            self.cost_new = np.zeros(self.particle.number_particles_all)
            self.reg_new = np.zeros(self.particle.number_particles_all)
            self.misfit_new = np.zeros(self.particle.number_particles_all)

            for m in range(self.particle.number_particles_all):
                # compute the old cost
                x = self.particle.particles[m]
                cost_old, reg_old, misfit_old = self.model.cost(x)
                self.cost_new[m], self.reg_new[m], self.misfit_new[m] = cost_old, reg_old, misfit_old

                line_search = False
                if line_search:
                    # do line search
                    descent = 0
                    x_star = self.model.generate_vector()
                    while descent == 0 and self.n_backtrack[m] < max_backtracking_iter:
                        # update the parameter
                        x_star[:] = 0.
                        x_star += x
                        x_star += self.alpha[m] * deltap[m]
                        # update the state at new parameter

                        # evaluate the cost functional, here the potential
                        self.cost_new[m], self.reg_new[m], self.misfit_new[m] = self.model.cost(x_star)

                        # Check if armijo conditions are satisfied
                        if m < self.particle.number_particles:
                            if (self.cost_new[m] < cost_old + self.alpha[m] * c_armijo * self.pg_phat[m]) or \
                                    (-self.pg_phat[m] <= self.options["gdm_tolerance"]):
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5
                                # print("alpha = ", alpha[m])
                        else:  # we do not have pg_phat for m >= particle.number_particles
                            if self.cost_new[m] < cost_old:
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5

            self.cost_mean, self.cost_std, self.cost_moment2 = np.mean(self.cost_new), np.std(self.cost_new), np.mean(
                self.cost_new ** 2)

            # compute the norm of the step/direction to move
            self.step_norm = self.pstep_norm * self.alpha

            # move all particles in the new directions, pm = pm + self.alpha[m] * sum_n phat[n] * k(pn, pm)
            self.particle.move(self.alpha, pstep)

            self.relative_grad_norm = np.divide(self.gradient_norm, self.gradient_norm_init)
            self.relative_step_norm = np.divide(self.step_norm, self.step_norm_init)

            self.time_computation += time.time() - time_computation

            # print data
            if print_level >= -1:
                if self.rank == 0:
                    print("\n{0:5} {1:5} {2:8} {3:15} {4:15} {5:15} {6:15} {7:15} {8:14}".format(
                        "it", "cpu", "id", "cost", "misfit", "reg", "||g||L2", "||m||L2", "alpha"))
                for m in range(self.particle.number_particles):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e} {8:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.relative_grad_norm[m], self.relative_step_norm[m], self.alpha[m]))
                for m in range(self.particle.number_particles, self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:15e} {4:15e} {5:15e} {6:15e} {7:14e}".format(
                        self.it, self.rank, m, self.cost_new[m], self.misfit_new[m], self.reg_new[m], 0.,
                        self.alpha[m]))

            # save data
            gradient_norm = np.empty([self.nproc, len(self.gradient_norm)], dtype=float)
            self.comm.Allgather(self.gradient_norm, gradient_norm)
            pg_phat = np.empty([self.nproc, len(self.pg_phat)], dtype=float)
            self.comm.Allgather(self.pg_phat, pg_phat)
            step_norm = np.empty([self.nproc, len(self.step_norm)], dtype=float)
            self.comm.Allgather(self.step_norm, step_norm)
            relative_grad_norm = np.empty([self.nproc, len(self.relative_grad_norm)], dtype=float)
            self.comm.Allgather(self.relative_grad_norm, relative_grad_norm)
            relative_step_norm = np.empty([self.nproc, len(self.relative_step_norm)], dtype=float)
            self.comm.Allgather(self.relative_step_norm, relative_step_norm)
            cost_new = np.empty([self.nproc, len(self.cost_new)], dtype=float)
            self.comm.Allgather(self.cost_new, cost_new)

            if self.rank == 0:
                self.data_save["gradient_norm"].append(np.mean(gradient_norm))
                # self.data_save["pg_phat"].append(pg_phat)
                self.data_save["step_norm"].append(np.mean(step_norm))
                # self.data_save["relative_grad_norm"].append(relative_grad_norm)
                # self.data_save["relative_step_norm"].append(relative_step_norm)
                # self.data_save["cost_new"].append(cost_new)
                self.data_save["dimension"].append(self.particle.dimension)
                self.data_save["cost_new"].append(cost_new)
                self.data_save["iteration"].append(self.it)
                self.data_save["meanL2norm"].append(self.meanL2norm)
                self.data_save["moment2L2norm"].append(self.moment2L2norm)
                self.data_save["meanErrorL2norm"].append(self.meanErrorL2norm)
                self.data_save["varianceL2norm"].append(self.varianceL2norm)
                self.data_save["varianceErrorL2norm"].append(self.varianceErrorL2norm)
                self.data_save["sample_trace"].append(self.sample_trace)
                self.data_save["cost_mean"].append(self.cost_mean)
                self.data_save["cost_std"].append(self.cost_std)
                self.data_save["cost_moment2"].append(self.cost_moment2)
                self.data_save["d"].append(self.variation.d)
                self.data_save["d_average"].append(self.variation.d_average_save)
                if self.model.qoi is not None:
                    self.data_save["qoi_mean"].append(self.qoi_mean)
                    self.data_save["qoi_std"].append(self.qoi_std)
                    self.data_save["qoi_moment2"].append(self.qoi_moment2)

                # N = self.nproc * self.particle.number_particles_all
                # filename = "data/data_nSamples_" + str(N) + "_isProjection_" + str(
                #     self.options["is_projection"]) + "_SVGD.p"

            # verify stopping criteria
            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if self.gradient_norm[m] > self.tol_gradient[m]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles_all):  # should use _all
                if self.n_backtrack[m] < max_backtracking_iter:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = False
                self.reason = 2
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if -self.pg_phat[m] > self.options["gdm_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 3
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                if self.step_norm[m] > self.step_norm_init[m] * self.options["step_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 4
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            # update data for optimization in next step
            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])

                self.save()

                if self.rank == 0:
                    pickle.dump(self.data_save, open(self.filename, 'wb'))
                break

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.gradient_norm = np.insert(self.gradient_norm, m - self.particle.number_particles_add, 0.)
                        self.gradient_norm_init = np.insert(self.gradient_norm_init,
                                                            m - self.particle.number_particles_add, 0.)
                        self.step_norm_init = np.insert(self.step_norm_init, m - self.particle.number_particles_add, 0.)
                        self.final_grad_norm = np.insert(self.final_grad_norm, m - self.particle.number_particles_add,
                                                         0.)

            # update variation, kernel, and hessian before solving the Newton linear system
            self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
            relative_step_norm = np.max(self.relative_step_norm)
            relative_step_norm_reduce = np.zeros(1, dtype=float)
            self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
            self.relative_step_norm = relative_step_norm_reduce[0]
            self.variation.update(self.particle, self.it, self.relative_step_norm)
            self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
            self.kernel.update(self.particle, self.variation)

            # save the particles for visualization and plot the eigenvalues at each particle
            self.save()

    # def solve_adam(self):
    #     # D.P. Kingma, J. Ba. Adam: A method for stochastic optimization. https://arxiv.org/pdf/1412.6980.pdf
    #     # to be tuned, it does not seem to work well
    #     rel_tol = self.options["rel_tolerance"]
    #     abs_tol = self.options["abs_tolerance"]
    #     max_iter = self.options["max_iter"]
    #     inner_tol = self.options["inner_rel_tolerance"]
    #     print_level = self.options["print_level"]
    #
    #     max_backtracking_iter = self.options["max_backtracking_iter"]
    #
    #     gradient_norm = np.zeros(self.particle.number_particles)
    #     gradient_norm_init = np.zeros(self.particle.number_particles)
    #     tol_gradient = np.zeros(self.particle.number_particles)
    #     cost_new = np.zeros(self.particle.number_particles)
    #     reg_new = np.zeros(self.particle.number_particles)
    #     misfit_new = np.zeros(self.particle.number_particles)
    #
    #     self.it = 0
    #     alpha = 1.e-3 * np.ones(self.particle.number_particles)
    #     beta_1 = 0.9
    #     beta_2 = 0.999
    #     mt_1 = [self.model.generate_vector() for m in range(self.particle.number_particles)]
    #     mt_2 = [self.model.generate_vector() for m in range(self.particle.number_particles)]
    #
    #     vt_1 = [self.model.generate_vector() for m in range(self.particle.number_particles)]
    #     vt_2 = [self.model.generate_vector() for m in range(self.particle.number_particles)]
    #
    #     epsilon = 1.e-8
    #
    #     while self.it < max_iter and (self.converged is False):
    #         # always update variation and kernel before compute the gradient
    #         self.variation.update(self.particle)
    #         self.kernel.update(self.particle, self.variation)
    #
    #         gt = [self.model.generate_vector() for m in range(self.particle.number_particles)]
    #         n_backtrack = np.zeros(self.particle.number_particles)
    #         for m in range(self.particle.number_particles):  # solve for each particle
    #             # evaluate gradient
    #             gradient = self.model.generate_vector()
    #             gradient_norm[m] = self.gradientSeparated(gradient, m)
    #
    #             gt[m].axpy(1.0, gradient)
    #
    #             if self.it == 0:
    #                 gradient_norm_init[m] = gradient_norm[m]
    #                 tol_gradient[m] = max(abs_tol, gradient_norm_init[m] * rel_tol)
    #
    #             mt_2[m].axpy(beta_1, mt_1[m])
    #             mt_2[m].axpy((1-beta_1), gt[m])
    #             mt_1[m].zero()
    #             mt_1[m].axpy(1./(1-np.power(beta_1, self.it+1)), mt_2[m])
    #
    #             vt_2[m].axpy(beta_2, vt_1[m])
    #             gt_array = gt[m]
    #             gt[m].set_local(gt_array**2)
    #             vt_2[m].axpy((1-beta_2), gt[m])
    #             vt_1[m].axpy(1./(1-np.power(beta_2, self.it+1)), vt_2[m])
    #
    #             mt_array = mt_1[m]
    #             vt_array = vt_1[m]
    #             gt[m].set_local(mt_array/(np.sqrt(vt_array) + epsilon))
    #
    #             # update particles
    #             self.particle.particles[m].axpy(-alpha[m], gt[m])
    #
    #             # update moments
    #             mt_1[m].zero()
    #             mt_1[m].axpy(1.0, mt_2[m])
    #             vt_1[m].zero()
    #             vt_1[m].axpy(1.0, vt_2[m])
    #
    #             # compute cost
    #             x_star = self.model.generate_vector()
    #             x_star[].zero()
    #             x_star[].axpy(1., self.particle.particles[m])
    #             x_star[STATE].zero()
    #             self.model.solveFwd(x_star[STATE], x_star, inner_tol)
    #
    #             cost_new[m], reg_new[m], misfit_new[m] = self.model.cost(x_star)
    #
    #         if (self.rank == 0) and (print_level >= -1):
    #             print("\n{0:5} {1:8} {2:15} {3:15} {4:15} {5:14} {6:14}".format(
    #                 "It", "id", "cost", "misfit", "reg", "||g||L2", "alpha"))
    #             for m in range(self.particle.number_particles):
    #                 print("{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e}".format(
    #                     self.it, m, cost_new[m], misfit_new[m], reg_new[m], gradient_norm[m], alpha[m]))
    #
    #         done = True
    #         for m in range(self.particle.number_particles):
    #             self.final_grad_norm[m] = gradient_norm[m]
    #             if gradient_norm[m] > tol_gradient[m]:
    #                 done = False
    #         if done:
    #             self.converged = True
    #             self.reason = 1
    #             break
    #
    #         done = True
    #         for m in range(self.particle.number_particles):
    #             if n_backtrack[m] < max_backtracking_iter:
    #                 done = False
    #         if done:
    #             self.converged = False
    #             self.reason = 2
    #             break
    #
    #         self.it += 1
