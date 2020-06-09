
from __future__ import absolute_import, division, print_function

import numpy as np
from ..modeling.variables import STATE, PARAMETER
from ..algorithms.cgsolverSteihaug import CGSolverSteihaug
import time
import pickle
from mpi4py import MPI


# ########################################  separated Newton system ###################################################
class HessianSeparated:
    # construct a lumped Hessian action to solve Newton linear system
    def __init__(self, model, particle, variation, kernel, options, comm):
        self.model = model
        self.particle = particle  # number of particles
        self.variation = variation
        self.kernel = kernel
        self.options = options
        self.save_kernel = options["save_kernel"]
        self.type_Hessian = options["type_Hessian"]
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.m = 0  # which particle
        self.ncalls = np.zeros(particle.number_particles_all, dtype=int)

    def init_vector(self, pvec, dim):
        if self.options["is_projection"]:
            pvec.init(self.particle.coefficient_dimension)
        else:
            self.model.prior.init_vector(pvec, dim)

    def update(self, particle, variation, kernel):
        # class assignment to be checked
        self.particle = particle
        self.variation = variation
        self.kernel = kernel

    def index(self, m):
        self.m = m
        self.variation.low_rank_Hessian = self.variation.low_rank_Hessian_hold
        self.variation.low_rank_Hessian_average = self.variation.low_rank_Hessian_average_hold
        self.variation.gauss_newton_approx = self.variation.gauss_newton_approx_hold

    def mult(self, phat, pout):

        pout.zero()
        R_coeff = 0.
        R_phat = self.particle.generate_vector()
        if self.options["is_projection"]:
            R_phat.axpy(1.0, phat)
        else:
            self.model.prior.R.mult(phat, R_phat)
        pout_misfit = self.particle.generate_vector()
        pout_kernel = self.particle.generate_vector()

        if self.save_kernel:
            kernel_value = self.kernel.value_set[self.m]
            kernel_gradient = self.kernel.gradient_set[self.m]
        else:
            kernel_value = self.kernel.values(self.m)
            kernel_gradient = self.kernel.gradients(self.m)

        if self.kernel.delta_kernel:
            hessian_phat = self.particle.generate_vector()

            self.variation.hessian(R_phat, hessian_phat, self.m, self.rank)
            pout_misfit.axpy(kernel_value[self.rank][self.m] ** 2, hessian_phat)

            pout_kernel.axpy(kernel_gradient[self.rank][self.m].inner(phat), kernel_gradient[self.rank][self.m])

            R_coeff += kernel_value[self.rank][self.m] ** 2

        elif self.type_Hessian is "diagonal":
            # loop over each particle to compute the expectation
            for p in range(self.nproc):
                for ie in range(self.particle.number_particles):  # take the expectation
                    hessian_phat = self.particle.generate_vector()

                    self.variation.hessian(R_phat, hessian_phat, ie, p)
                    pout_misfit.axpy(kernel_value[p][ie] ** 2, hessian_phat)

                    pout_kernel.axpy(kernel_gradient[p][ie].inner(phat), kernel_gradient[p][ie])

                    R_coeff += kernel_value[p][ie] ** 2

            # also use the particle to compute the expectation
            if self.m >= self.particle.number_particles:
                hessian_phat = self.particle.generate_vector()

                self.variation.hessian(R_phat, hessian_phat, self.m, self.rank)
                pout_misfit.axpy(kernel_value[self.rank][self.m] ** 2, hessian_phat)

                pout_kernel.axpy(kernel_gradient[self.rank][self.m].inner(phat), kernel_gradient[self.rank][self.m])

                R_coeff += kernel_value[self.rank][self.m] ** 2

        elif self.type_Hessian is "lumped":
            for p in range(self.nproc):
                # loop over each particle to compute the expectation
                for ie in range(self.particle.number_particles):  # take the expectation
                    hessian_phat = self.particle.generate_vector()

                    self.variation.hessian(R_phat, hessian_phat, ie, p)
                    pout_misfit.axpy(self.kernel.value_sum_reduce[p][ie] * kernel_value[p][ie], hessian_phat)

                    pout_kernel.axpy(kernel_gradient[p][ie].inner(phat), self.kernel.gradient_sum_reduce[p][ie])

                    R_coeff += self.kernel.value_sum_reduce[p][ie] * kernel_value[p][ie]

                    if self.m >= self.particle.number_particles:
                        pout_misfit.axpy(kernel_value[p][ie] ** 2, hessian_phat)
                        pout_kernel.axpy(kernel_gradient[p][ie].inner(phat), kernel_gradient[p][ie])
                        R_coeff += kernel_value[p][ie] ** 2

            # also use the particle to compute the expectation
            if self.m >= self.particle.number_particles:
                hessian_phat = self.particle.generate_vector()
                self.variation.hessian(R_phat, hessian_phat, self.m, self.rank)
                pout_kernel.axpy(self.kernel.value_sum_reduce[self.rank][self.m] * kernel_value[self.rank][self.m], hessian_phat)
                pout_misfit.axpy(kernel_gradient[self.rank][self.m].inner(phat), self.kernel.gradient_sum_reduce[self.rank][self.m])
                R_coeff += self.kernel.value_sum_reduce[self.rank][self.m] * kernel_value[self.rank][self.m]

                pout_misfit.axpy(kernel_value[self.rank][self.m] ** 2, hessian_phat)
                pout_kernel.axpy(kernel_gradient[self.rank][self.m].inner(phat), kernel_gradient[self.rank][self.m])
                R_coeff += kernel_value[self.rank][self.m] ** 2

        else:
            raise NotImplementedError("choose between diagonal and lumped")

        phelp = self.particle.generate_vector()
        if not self.options["is_projection"]:
            self.model.prior.R.mult(pout_misfit, phelp)
        pout_misfit.zero()
        pout_misfit.axpy(1.0, phelp)

        pout.axpy(1.0, pout_misfit)
        pout.axpy(1.0, pout_kernel)
        pout.axpy(R_coeff, R_phat)

        self.ncalls[self.m] += 1

    def assemble(self):
        if self.save_kernel:
            kernel_value = self.kernel.value_set[self.m]
            kernel_gradient = self.kernel.gradient_set[self.m]
        else:
            kernel_value = self.kernel.values(self.m)
            kernel_gradient = self.kernel.gradients(self.m)

        hess = np.zeros((self.particle.coefficient_dimension, self.particle.coefficient_dimension))

        if self.kernel.delta_kernel:
            hess += self.variation.hessian_misfit_gather[self.rank][self.m] + np.eye(self.particle.coefficient_dimension)
            hess *= (kernel_value[self.rank][self.m] ** 2)
            hess += np.outer(kernel_gradient[self.rank][self.m].get_local(), kernel_gradient[self.rank][self.m].get_local())

        elif self.type_Hessian is "diagonal":
            # loop over each particle to compute the expectation
            for p in range(self.nproc):
                for ie in range(self.particle.number_particles):  # take the expectation
                    hess += kernel_value[p][ie] ** 2 * (self.variation.hessian_misfit_gather[p][ie]
                                                        + np.eye(self.particle.coefficient_dimension))
                    hess += np.outer(kernel_gradient[p][ie].get_local(), kernel_gradient[p][ie].get_local())
                if self.m > self.particle.number_particles:
                    hess += kernel_value[p][self.m] ** 2 * (self.variation.hessian_misfit_gather[p][self.m]
                                                            + np.eye(self.particle.coefficient_dimension))
                    hess += np.outer(kernel_gradient[p][self.m].get_local(), kernel_gradient[p][self.m].get_local())

        elif self.type_Hessian is "lumped":
            # loop over each particle to compute the expectation
            for p in range(self.nproc):
                for ie in range(self.particle.number_particles):  # take the expectation
                    hess += kernel_value[p][ie] * self.kernel.value_sum_reduce[p][ie] * (self.variation.hessian_misfit_gather[p][ie]
                                                        + np.eye(self.particle.coefficient_dimension))
                    hess += np.outer(self.kernel.gradient_sum_reduce[p][ie].get_local(), kernel_gradient[p][ie].get_local())
                    if self.m > self.particle.number_particles:
                        hess += kernel_value[p][ie] ** 2 * (self.variation.hessian_misfit_gather[p][ie]
                                                        + np.eye(self.particle.coefficient_dimension))
                        hess += np.outer(kernel_gradient[p][ie].get_local(), kernel_gradient[p][ie].get_local())
                if self.m > self.particle.number_particles:
                    hess += kernel_value[p][self.m] * self.kernel.value_sum_reduce[p][self.m] * (
                                self.variation.hessian_misfit_gather[p][self.m]
                                + np.eye(self.particle.coefficient_dimension))
                    hess += np.outer(self.kernel.gradient_sum_reduce[p][self.m].get_local(),
                                     kernel_gradient[p][self.m].get_local())
                    hess += kernel_value[p][self.m] ** 2 * (self.variation.hessian_misfit_gather[p][self.m]
                                                            + np.eye(self.particle.coefficient_dimension))
                    hess += np.outer(kernel_gradient[p][self.m].get_local(), kernel_gradient[p][self.m].get_local())

        return hess


class PreconditionerSeparated:

    def __init__(self, model, options):
        self.model = model
        self.options = options

    def solve(self, phat, prhs):
        phat.zero()
        if self.options["is_projection"]:
            phat.axpy(1.0, prhs)
        else:
            self.model.prior.Rsolver.solve(phat, prhs)


class NewtonSeparated:
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
        self.tol_newton = np.zeros(self.particle.number_particles_all)
        self.tol_cg = np.zeros(self.particle.number_particles_all)
        self.total_cg_iter = np.zeros(self.particle.number_particles_all)
        self.final_grad_norm = np.zeros(self.particle.number_particles_all)
        self.cost_new = np.zeros(self.particle.number_particles_all)
        self.reg_new = np.zeros(self.particle.number_particles_all)
        self.misfit_new = np.zeros(self.particle.number_particles_all)
        self.alpha = np.ones(self.particle.number_particles_all)
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
            gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
            gradient.axpy(kernel_value[self.rank][m], gp_misfit)
            gradient.axpy(-1.0, kernel_gradient[self.rank][m])
        else:
            for p in range(self.nproc):
                for ie in range(self.particle.number_particles):  # take the expectation over particle set
                    gp_misfit = self.particle.generate_vector()
                    gp_misfit.set_local(self.variation.gradient_gather[p][ie].get_local())
                    gradient.axpy(kernel_value[p][ie], gp_misfit)
                    gradient.axpy(-1.0, kernel_gradient[p][ie])

            # also use the particle to compute the expectation
            if m >= self.particle.number_particles:
                gp_misfit = self.particle.generate_vector()
                gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
                gradient.axpy(kernel_value[self.rank][m], gp_misfit)
                gradient.axpy(-1.0, kernel_gradient[self.rank][m])

        if self.options["is_projection"]:
            gradient_norm = np.sqrt(gradient.inner(gradient))
        else:
            tmp = self.particle.generate_vector()
            self.model.prior.Msolver.solve(tmp, gradient)
            gradient_norm = np.sqrt(gradient.inner(tmp))

        return gradient_norm

    def communication(self, phat):
        time_communication = time.time()

        phat_array = np.empty([self.particle.number_particles_all, self.particle.dimension], dtype=float)
        phat_gather_array = np.empty([self.nproc, self.particle.number_particles_all, self.particle.dimension], dtype=float)
        for n in range(self.particle.number_particles_all):
            phat_array[n, :] = phat[n].get_local()
        self.comm.Allgather(phat_array, phat_gather_array)

        phat_gather = [[self.particle.generate_vector() for n in range(self.particle.number_particles_all)] for p
                       in range(self.nproc)]
        for p in range(self.nproc):
            for n in range(self.particle.number_particles_all):
                phat_gather[p][n].set_local(phat_gather_array[p, n, :])

        self.time_communication += time.time() - time_communication

        return phat_gather

    def solve(self):
        # use Newton method to solve the optimization problem
        rel_tol = self.options["rel_tolerance"]
        abs_tol = self.options["abs_tolerance"]
        max_iter = self.options["max_iter"]
        line_search = self.options["line_search"]
        inner_tol = self.options["inner_rel_tolerance"]
        print_level = self.options["print_level"]
        cg_coarse_tolerance = self.options["cg_coarse_tolerance"]
        c_armijo = self.options["c_armijo"]
        max_backtracking_iter = self.options["max_backtracking_iter"]

        hessian = HessianSeparated(self.model, self.particle, self.variation, self.kernel, self.options, self.comm)
        preconditioner = PreconditionerSeparated(self.model, self.options)
        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)
        hessian.update(self.particle, self.variation, self.kernel)

        self.it = 0
        self.converged = False

        if self.save_number:
            self.kernel.save_values(self.save_number, self.it)
            self.particle.save(self.save_number, self.it)
            self.variation.save_eigenvalue(self.save_number, self.it)
            if self.plot:
                self.variation.plot_eigenvalue(self.save_number, self.it)

        self.cost_mean, self.cost_std = 0., 0.

        while self.it < max_iter and (self.converged is False):

            if self.options["type_parameter"] is 'vector' and self.plot:
                self.particle.plot_particles(self.particle, self.it)

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

            phat = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            self.pg_phat = np.zeros(self.particle.number_particles_all)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            t0 = time.time()
            for m in range(self.particle.number_particles_all):  # solve for each particle used for constructing the map
                # evaluate gradient
                gradient = self.particle.generate_vector()
                self.gradient_norm[m] = self.gradientSeparated(gradient, m)
                # print("self.gradient_norm[m]", self.gradient_norm[m])

                # set tolerance for Newton iteration
                if self.it == 0:   # or self.variation.is_bases_updated:
                    self.gradient_norm_init[m] = self.gradient_norm[m]
                    self.tol_newton[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    if m >= self.particle.number_particles_old \
                            and m < self.particle.number_particles_all - self.particle.number_particles_add:
                        self.gradient_norm_init[m] = self.gradient_norm[m]
                        self.tol_newton[m] = max(abs_tol, self.gradient_norm_init[m] * rel_tol)

                # set tolerance for CG iteration
                self.tol_cg[m] = min(cg_coarse_tolerance,
                                     np.sqrt(self.gradient_norm[m] / self.gradient_norm_init[m]))

                hessian.index(m)  # which particle to deal with

                if self.options["is_projection"]:  # direct solve
                    hess_array = hessian.assemble()
                    # print("hess", hess)
                    # d, U = np.linalg.eig(hess)
                    # print("d", d)
                    grad_array = gradient.get_local()
                    phat_array = np.linalg.solve(hess_array, -grad_array)
                    phat[m].set_local(phat_array)
                    # print("phat_array", phat_array)

                else:  # CG solve
                    # set Hessian and CG solver
                    solver = CGSolverSteihaug(comm=self.model.prior.R.mpi_comm())
                    solver.set_operator(hessian)
                    solver.set_preconditioner(preconditioner)
                    solver.parameters["rel_tolerance"] = self.tol_cg[m]
                    solver.parameters["zero_initial_guess"] = True
                    solver.parameters["print_level"] = print_level
                    solver.parameters["max_iter"] = 20

                    # solve the Newton linear system
                    solver.solve(phat[m], -gradient)

                    # # replace Hessian with its Gauss Newton approximation if CG reached a negative direction
                    # if solver.reasonid == 2:
                    #     hessian.variation.low_rank_Hessian = False
                    #     hessian.variation.gauss_newton_approx = True
                    #     solver.set_operator(hessian)
                    #     solver.solve(phat[m], -gradient)

                self.total_cg_iter[m] += hessian.ncalls[m]
                self.pg_phat[m] = gradient.inner(phat[m])

            t1 = time.time()-t0
            if self.options["print_level"] > 2:
                print("time to solve " + str(self.particle.number_particles_all) + " Newton linear systems = ", t1)

            phat = self.communication(phat)  # gather the coefficients to all processors

            # step for particle update, pstep(x) = sum_n phat_n k_n(x)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            deltap = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles_all)]
            for m in range(self.particle.number_particles_all):
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles):
                        pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat[p][n])
                    if m >= self.particle.number_particles:
                        pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat[p][m])

                if self.options["is_projection"]:
                    self.pstep_norm[m] = np.sqrt(pstep[m].inner(pstep[m]))/self.particle.coefficient_dimension
                    pstep_m = pstep[m].get_local()
                    for r in range(self.particle.coefficient_dimension):
                        deltap[m].axpy(pstep_m[r], self.particle.bases[r])
                else:
                    phelp = self.model.generate_vector(PARAMETER)
                    self.model.prior.M.mult(pstep[m], phelp)
                    self.pstep_norm[m] = np.sqrt(pstep[m].inner(phelp))
                    deltap[m].axpy(1.0, pstep[m])

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    if m >= self.particle.number_particles_old \
                            and m < self.particle.number_particles_all - self.particle.number_particles_add:
                        self.step_norm_init[m] = self.pstep_norm[m]

            if self.it == 0:
                self.step_norm_init = self.pstep_norm

            # line search
            self.alpha = np.ones(self.particle.number_particles_all)
            self.n_backtrack = np.zeros(self.particle.number_particles_all, dtype=int)
            self.cost_new = np.zeros(self.particle.number_particles_all)
            self.reg_new = np.zeros(self.particle.number_particles_all)
            self.misfit_new = np.zeros(self.particle.number_particles_all)

            for m in range(self.particle.number_particles_all):
                # compute the old cost
                x = self.variation.x_all[m]
                cost_old, reg_old, misfit_old = self.model.cost(x)
                self.cost_new[m], self.reg_new[m], self.misfit_new[m] = cost_old, reg_old, misfit_old

                # print("self.cost_new[m], self.reg_new[m], self.misfit_new[m]", self.cost_new[m], self.reg_new[m], self.misfit_new[m])

                if line_search:
                    # do line search
                    descent = 0
                    x_star = self.model.generate_vector()
                    while descent == 0 and self.n_backtrack[m] < max_backtracking_iter:
                        # update the parameter
                        x_star[PARAMETER].zero()
                        x_star[PARAMETER].axpy(1., x[PARAMETER])
                        x_star[PARAMETER].axpy(self.alpha[m], deltap[m])
                        # update the state at new parameter
                        x_star[STATE].zero()
                        x_star[STATE].axpy(1., x[STATE])
                        self.model.solveFwd(x_star[STATE], x_star, inner_tol)

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
                        else:  # we do not have pg_phat for m >= particle.number_particles
                            if self.cost_new[m] < cost_old:
                                cost_old = self.cost_new[m]
                                descent = 1
                            else:
                                self.n_backtrack[m] += 1
                                self.alpha[m] *= 0.5
                        # print("rank = ", self.rank, "m = ", m, "alpha[m] = ", self.alpha[m], "self.n_backtrack[m]", self.n_backtrack[m])
                # print("rank = ", self.rank, "m = ", m, "alpha[m] = ", self.alpha[m], "self.n_backtrack[m]", self.n_backtrack[m])

            self.cost_mean, self.cost_std, self.cost_moment2 = np.mean(self.cost_new), np.std(self.cost_new), np.mean(self.cost_new**2)

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
                    print("\n{0:4} {1:4} {2:4} {3:5} {4:15} {5:15} {6:15} {7:15} {8:14} {9:14} {10:14} {11:14}".format(
                         "it", "cpu", "m", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "||dm||L2", "alpha", "tolcg"))
                for m in range(self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:4d} {4:15e} {5:15e} {6:15e} {7:15e} {8:14e} {9:14e} {10:14e} {11:14e}".format(
                        self.it, self.rank, m, hessian.ncalls[m], self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.pg_phat[m], self.relative_grad_norm[m], self.relative_step_norm[m], self.alpha[m], self.tol_cg[m]))

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
                filename = "data/data_nSamples_"+str(N)+"_isProjection_"+str(self.options["is_projection"])+"_SVN.p"

            # verify stopping criteria
            done = True
            for m in range(self.particle.number_particles):
                self.final_grad_norm[m] = self.gradient_norm[m]
                if self.gradient_norm[m] > self.tol_newton[m]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
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
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            done = True
            for m in range(self.particle.number_particles):
                if -self.pg_phat[m] > self.options["gdm_tolerance"]:
                    done = False
            done_gather = self.comm.allgather(done)
            if np.sum(done_gather) == self.nproc:
                self.converged = True
                self.reason = 3
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
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
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                # self.options["is_projection"] = False
                break

            # update data for optimization in next step
            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)

                if self.rank == 0:
                    pickle.dump(self.data_save, open(filename, 'wb'))
                break

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                # expand the arrays if new particles are added
                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.gradient_norm = np.insert(self.gradient_norm, m-self.particle.number_particles_add, 0.)
                        self.gradient_norm_init = np.insert(self.gradient_norm_init, m-self.particle.number_particles_add, 0.)
                        self.step_norm_init = np.insert(self.step_norm_init, m-self.particle.number_particles_add, 0.)
                        self.tol_newton = np.insert(self.tol_newton, m-self.particle.number_particles_add, 0)
                        self.tol_cg = np.insert(self.tol_cg, m-self.particle.number_particles_add, 0.)
                        self.total_cg_iter = np.insert(self.total_cg_iter, m-self.particle.number_particles_add, 0)
                        self.final_grad_norm = np.insert(self.final_grad_norm, m-self.particle.number_particles_add, 0.)
                        hessian.ncalls = np.insert(hessian.ncalls, m - self.particle.number_particles_add, 0)

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
            hessian.update(self.particle, self.variation, self.kernel)

            # save the particles for visualization and plot the eigenvalues at each particle
            if self.save_number and np.mod(self.it, self.save_step) == 0:
                self.kernel.save_values(self.save_number, self.it)
                self.particle.save(self.save_number, self.it)
                self.variation.save_eigenvalue(self.save_number, self.it)
                if self.plot:
                    self.variation.plot_eigenvalue(self.save_number, self.it)
