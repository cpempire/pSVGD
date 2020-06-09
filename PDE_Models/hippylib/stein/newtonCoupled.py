# contact: Peng Chen, peng@ices.utexas.edu,
# written on Novemeber 19, 2018
# updated for parallel projection version on Jan 12, 2019

from __future__ import absolute_import, division, print_function

import dolfin as dl
import numpy as np
from ..modeling.variables import STATE, PARAMETER
from ..algorithms.cgsolverSteihaug import CGSolverSteihaug
from ..utils.vector2function import vector2Function
from mpi4py import MPI


# ########################################  coupled Newton system ###################################################
class GlobalLocalSwitch:
    # a class to map a list of local vectors to a global vector and back
    def __init__(self, model, particle, options, comm):
        self.particle = particle
        self.model = model
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.functionspace()

    def update(self, particle):
        self.particle = particle
        self.functionspace()

    def functionspace(self):
        if self.options["is_projection"]:
            dim_Local = self.particle.dimension
            dim_Global = self.particle.dimension*self.particle.number_particles_all
            self.Vh_Global = dl.VectorFunctionSpace(self.particle.Vh_COEFF.mesh(), "R", degree=0,
                                                    dim=dim_Global)
            self.Vh_Local = dl.VectorFunctionSpace(self.particle.Vh_COEFF.mesh(), "R", degree=0,
                                                   dim=dim_Local)
            # dim_Universal = self.nproc*self.particle.dimension*self.particle.number_particles_all
            # self.Vh_Universal = dl.VectorFunctionSpace(self.particle.Vh_COEFF.mesh(), "R", degree=0,
            #                                         dim=dim_Universal)

        else:
            if self.options["type_parameter"] is 'field':
                if self.particle.number_particles_all == 1:
                    self.Vh_Global = dl.FunctionSpace(self.model.problem.Vh[PARAMETER].mesh(),
                                  'Lagrange', self.model.problem.Vh[PARAMETER].ufl_element().degree())
                    self.Vh_Local = dl.FunctionSpace(self.model.problem.Vh[PARAMETER].mesh(),
                                  'Lagrange', self.model.problem.Vh[PARAMETER].ufl_element().degree())
                    # if self.nproc == 1:
                    #     self.Vh_Universal = dl.FunctionSpace(self.model.problem.Vh[PARAMETER].mesh(),
                    #                                       'Lagrange',
                    #                                       self.model.problem.Vh[PARAMETER].ufl_element().degree())
                    # else:
                    #     self.Vh_Universal = dl.VectorFunctionSpace(self.model.problem.Vh[PARAMETER].mesh(),
                    #                            'Lagrange', self.model.problem.Vh[PARAMETER].ufl_element().degree(),
                    #                            dim=self.nproc)
                else:
                    self.Vh_Global = dl.VectorFunctionSpace(self.model.problem.Vh[PARAMETER].mesh(),
                                                             'Lagrange', self.model.problem.Vh[PARAMETER].ufl_element().degree(),
                                                             dim=self.particle.number_particles_all)
                    self.Vh_Local = self.Vh_Global.sub(0).collapse()
                    # self.Vh_Universal = dl.VectorFunctionSpace(self.model.problem.Vh[PARAMETER].mesh(),
                    #                        'Lagrange', self.model.problem.Vh[PARAMETER].ufl_element().degree(),
                    #                        dim=self.nproc*self.particle.number_particles_all)

                    self.local_spaces = [self.Vh_Local for m in range(self.particle.number_particles_all)]
                    self.global_space = self.Vh_Global
                    self.to_sub = dl.FunctionAssigner(self.local_spaces, self.global_space)
                    self.from_sub = dl.FunctionAssigner(self.global_space, self.local_spaces)
            elif self.options["type_parameter"] is 'vector':
                dim_Local = self.model.prior.dim
                dim_Global = self.model.prior.dim*self.particle.number_particles_all
                self.Vh_Global = dl.VectorFunctionSpace(self.model.problem.Vh[PARAMETER].mesh(), "R", degree=0,
                                                        dim=dim_Global)
                self.Vh_Local = dl.VectorFunctionSpace(self.model.problem.Vh[PARAMETER].mesh(), "R", degree=0,
                                                       dim=dim_Local)
                # dim_Universal = self.nproc*self.model.prior.dim*self.particle.number_particles_all
                # self.Vh_Universal = dl.VectorFunctionSpace(self.model.problem.Vh[PARAMETER].mesh(), "R", degree=0,
                #                                         dim=dim_Universal)

    def global2local(self, p_global, p_local):

        if self.particle.number_particles_all == 1:
            p_local[0].zero()
            p_local[0].axpy(1.0, p_global)
        else:
            if self.options["is_projection"]:
                p_global_vec = p_global.get_local()
                for n in range(self.particle.number_particles_all):
                    p_local[n].zero()
                    p_local[n].set_local(p_global_vec[n * self.particle.dimension:(n + 1) * self.particle.dimension])
            else:
                if self.options["type_parameter"] is 'field':
                    p_global_fun = vector2Function(p_global, self.Vh_Global)
                    p_local_funs = [dl.Function(self.Vh_Local) for m in range(self.particle.number_particles_all)]
                    self.to_sub.assign(p_local_funs, p_global_fun)
                    for n in range(self.particle.number_particles_all):
                        p_local[n].zero()
                        p_local[n].axpy(1.0, p_local_funs[n].vector())
                elif self.options["type_parameter"] is 'vector':
                    p_global_vec = p_global.get_local()
                    for n in range(self.particle.number_particles_all):
                        p_local[n].zero()
                        p_local[n].set_local(p_global_vec[n*self.model.prior.dim:(n+1)*self.model.prior.dim])

    def local2global(self, p_local, p_global):
        if self.particle.number_particles_all == 1:
            p_global.zero()
            p_global.axpy(1.0, p_local[0])
        else:
            if self.options["is_projection"]:
                p_global_vec = p_global.get_local()
                for n in range(self.particle.number_particles_all):
                    p_global_vec[n*self.particle.dimension:(n+1)*self.particle.dimension] = p_local[n].get_local()
                p_global.set_local(p_global_vec)
            else:
                if self.options["type_parameter"] is 'field':
                    p_global.zero()
                    p_global_fun = vector2Function(p_global, self.Vh_Global)
                    p_local_funs = [vector2Function(p_local[n], self.Vh_Local) for n in range(self.particle.number_particles_all)]
                    self.from_sub.assign(p_global_fun, p_local_funs)
                    p_global.zero()
                    p_global.axpy(1.0, p_global_fun.vector())
                elif self.options["type_parameter"] is 'vector':
                    p_global_vec = p_global.get_local()
                    for n in range(self.particle.number_particles_all):
                        p_global_vec[n*self.model.prior.dim:(n+1)*self.model.prior.dim] = p_local[n].get_local()
                    p_global.set_local(p_global_vec)


class PreconditionerCoupled:
    # define the preconditioner for the coupled Newton system
    def __init__(self, model, particle, options, comm):
        self.model = model
        self.particle = particle
        self.options = options
        self.comm = comm
        self.gls = GlobalLocalSwitch(self.model, self.particle, self.options, self.comm)

    def update(self, particle):
        self.particle = particle
        self.gls.update(particle)

    def solve(self, phat, prhs):

        prhs_subs = [self.model.generate_vector(PARAMETER) for n in range(self.particle.number_particles_all)]
        phat_subs = [self.model.generate_vector(PARAMETER) for n in range(self.particle.number_particles_all)]

        self.gls.global2local(prhs, prhs_subs)

        for m in range(self.particle.number_particles_all):
            self.model.prior.Rsolver.solve(phat_subs[m], prhs_subs[m])

        self.gls.local2global(phat_subs, phat)


class HessianCoupled:
    # construct Hessian action in a vector of length parameter_dimension * number_particles
    def __init__(self, model, particle, variation, kernel, options, comm):
        self.model = model
        self.particle = particle  # number of particles
        self.variation = variation
        self.kernel = kernel
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.save_kernel = options["save_kernel"]
        self.type_Hessian = options["type_Hessian"]
        self.ncalls = 0
        self.gls = GlobalLocalSwitch(self.model, self.particle, self.options, self.comm)

        if not self.options["is_projection"]:
            if self.particle.number_particles_all == 1:
                z_trial = dl.TrialFunction(self.gls.Vh_Local)
                z_test = dl.TestFunction(self.gls.Vh_Local)
                self.Hessian = dl.assemble(z_trial*z_test*dl.dx)
            else:
                z_vec = [None] * particle.number_particles_all
                c_vec = [None] * particle.number_particles_all
                for n in range(particle.number_particles_all):
                    z_trial = dl.TrialFunction(self.gls.Vh_Global.sub(0))
                    z_test = dl.TestFunction(self.gls.Vh_Global.sub(0))
                    z_vec[n] = z_trial
                    c_vec[n] = z_test
                z_vec = dl.as_vector(z_vec)
                c_vec = dl.as_vector(c_vec)
                self.Hessian = dl.assemble(dl.dot(c_vec, z_vec) * dl.dx)

    def init_vector(self, pvec, dim):
        self.Hessian.init_vector(pvec, dim)

    def update(self, particle, variation, kernel):
        self.particle = particle
        self.variation = variation
        self.kernel = kernel
        self.gls.update(particle)

        if not self.options["is_projection"]:
            if self.particle.number_particles_all == 1:
                z_trial = dl.TrialFunction(self.gls.Vh_Local)
                z_test = dl.TestFunction(self.gls.Vh_Local)
                self.Hessian = dl.assemble(z_trial*z_test*dl.dx)
            else:
                z_vec = [None] * particle.number_particles_all
                c_vec = [None] * particle.number_particles_all
                for n in range(particle.number_particles_all):
                    z_trial = dl.TrialFunction(self.gls.Vh_Global.sub(0))
                    z_test = dl.TestFunction(self.gls.Vh_Global.sub(0))
                    z_vec[n] = z_trial
                    c_vec[n] = z_test
                z_vec = dl.as_vector(z_vec)
                c_vec = dl.as_vector(c_vec)
                self.Hessian = dl.assemble(dl.dot(c_vec, z_vec) * dl.dx)

    def mult(self, phat, pout):

        pout_subs = [self.model.generate_vector(PARAMETER) for n in range(self.particle.number_particles_all)]
        phat_subs = [self.model.generate_vector(PARAMETER) for n in range(self.particle.number_particles_all)]

        self.gls.global2local(phat, phat_subs)

        for m in range(self.particle.number_particles_all):

            R_coeff = 0.
            R_phat = self.particle.generate_vector()
            if self.options["is_projection"]:
                R_phat.axpy(1.0, phat_subs[m])
            else:
                self.model.prior.R.mult(phat_subs[m], R_phat)
            pout_misfit = self.particle.generate_vector()
            pout_kernel = self.particle.generate_vector()

            if self.save_kernel:
                kernel_value_m = self.kernel.value_set[m]
                kernel_gradient_m = self.kernel.gradient_set[m]
            else:
                kernel_value_m = self.kernel.values(m)
                kernel_gradient_m = self.kernel.gradients(m)

            if self.kernel.delta_kernel:
                hessian_phat = self.particle.generate_vector()
                self.variation.hessian(R_phat, hessian_phat, m, self.rank)
                pout_misfit.axpy(kernel_value_m[self.rank][m] ** 2, hessian_phat)
                pout_kernel.axpy(kernel_gradient_m[self.rank][m].inner(phat_subs[m]), kernel_gradient_m[self.rank][m])
                R_coeff += kernel_value_m[self.rank][m] ** 2

                phelp = self.particle.generate_vector()
                if not self.options["is_projection"]:
                    self.model.prior.R.mult(pout_misfit, phelp)
                pout_misfit.zero()
                pout_misfit.axpy(1.0, phelp)

                pout_subs[m].axpy(1.0, pout_misfit)
                pout_subs[m].axpy(1.0, pout_kernel)
                pout_subs[m].axpy(R_coeff, R_phat)

            elif self.type_Hessian is "diagonal":
                # loop over each particle to compute the expectation
                for p in range(self.nproc):
                    for ie in range(self.particle.number_particles):  # take the expectation
                        hessian_phat = self.particle.generate_vector()
                        self.variation.hessian(R_phat, hessian_phat, ie, p)
                        pout_misfit.axpy(kernel_value_m[p][ie] ** 2, hessian_phat)
                        pout_kernel.axpy(kernel_gradient_m[p][ie].inner(phat_subs[m]), kernel_gradient_m[p][ie])
                        R_coeff += kernel_value_m[p][ie] ** 2

                # also use the particle to compute the expectation
                if m >= self.particle.number_particles:
                    hessian_phat = self.particle.generate_vector()
                    self.variation.hessian(R_phat, hessian_phat, m, self.rank)
                    pout_misfit.axpy(kernel_value_m[self.rank][m] ** 2, hessian_phat)
                    pout_kernel.axpy(kernel_gradient_m[self.rank][m].inner(phat_subs[m]), kernel_gradient_m[self.rank][m])
                    R_coeff += kernel_value_m[self.rank][m] ** 2

                phelp = self.particle.generate_vector()
                if not self.options["is_projection"]:
                    self.model.prior.R.mult(pout_misfit, phelp)
                pout_misfit.zero()
                pout_misfit.axpy(1.0, phelp)

                pout_subs[m].axpy(1.0, pout_misfit)
                pout_subs[m].axpy(1.0, pout_kernel)
                pout_subs[m].axpy(R_coeff, R_phat)

            elif self.type_Hessian is "lumped":
                for p in range(self.nproc):
                    # loop over each particle to compute the expectation
                    for ie in range(self.particle.number_particles):  # take the expectation
                        hessian_phat = self.particle.generate_vector()
                        self.variation.hessian(R_phat, hessian_phat, ie, p)
                        pout_misfit.axpy(self.kernel.value_sum_reduce[p][ie] * kernel_value_m[p][ie], hessian_phat)
                        pout_kernel.axpy(kernel_gradient_m[p][ie].inner(phat_subs[m]), self.kernel.gradient_sum_reduce[p][ie])
                        R_coeff += self.kernel.value_sum_reduce[p][ie] * kernel_value_m[p][ie]

                        if m >= self.particle.number_particles:
                            pout_misfit.axpy(kernel_value_m[p][ie] ** 2, hessian_phat)
                            pout_kernel.axpy(kernel_gradient_m[p][ie].inner(phat_subs[m]), kernel_gradient_m[p][ie])
                            R_coeff += kernel_value_m[p][ie] ** 2

                # also use the particle to compute the expectation
                if m >= self.particle.number_particles:
                    hessian_phat = self.particle.generate_vector()
                    self.variation.hessian(R_phat, hessian_phat, m, self.rank)
                    pout_kernel.axpy(self.kernel.value_sum_reduce[self.rank][m] * kernel_value_m[self.rank][m],
                                     hessian_phat)
                    pout_misfit.axpy(kernel_gradient_m[self.rank][m].inner(phat),
                                     self.kernel.gradient_sum_reduce[self.rank][m])
                    R_coeff += self.kernel.value_sum_reduce[self.rank][m] * kernel_value_m[self.rank][m]

                    pout_misfit.axpy(kernel_value_m[self.rank][m] ** 2, hessian_phat)
                    pout_kernel.axpy(kernel_gradient_m[self.rank][m].inner(phat_subs[m]), kernel_gradient_m[self.rank][m])
                    R_coeff += kernel_value_m[self.rank][m] ** 2

                phelp = self.particle.generate_vector()
                if not self.options["is_projection"]:
                    self.model.prior.R.mult(pout_misfit, phelp)
                pout_misfit.zero()
                pout_misfit.axpy(1.0, phelp)

                pout_subs[m].axpy(1.0, pout_misfit)
                pout_subs[m].axpy(1.0, pout_kernel)
                pout_subs[m].axpy(R_coeff, R_phat)

            elif self.type_Hessian is "full":  # assemble the full coupled Hessian system
                # loop for each Hessian H_mn for n = 1, ..., number_particles
                for n in range(self.particle.number_particles):

                    if self.save_kernel:
                        kernel_value_n = self.kernel.value_set[n]
                        kernel_gradient_n = self.kernel.gradient_set[n]
                    else:
                        kernel_value_n = self.kernel.values(n)
                        kernel_gradient_n = self.kernel.gradients(n)

                    pout_misfit.zero()
                    pout_kernel.zero()
                    R_coeff = 0.
                    R_phat.zero()
                    if self.options["is_projection"]:
                        R_phat.axpy(1.0, phat_subs[n])
                    else:
                        self.model.prior.R.mult(phat_subs[n], R_phat)

                    for p in range(self.nproc):
                        # loop over each particle used for map construction to compute the expectation
                        for ie in range(self.particle.number_particles):
                            hessian_phat = self.particle.generate_vector()
                            self.variation.hessian(R_phat, hessian_phat, ie, p)
                            pout_misfit.axpy(kernel_value_n[p][ie] * kernel_value_m[p][ie], hessian_phat)
                            pout_kernel.axpy(kernel_gradient_m[p][ie].inner(phat_subs[n]),
                                             kernel_gradient_n[p][ie])
                            R_coeff += kernel_value_n[p][ie] * kernel_value_m[p][ie]

                    # also use the particle to compute the expectation
                    if m >= self.particle.number_particles:
                        hessian_phat = self.particle.generate_vector()
                        self.variation.hessian(R_phat, hessian_phat, m, self.rank)
                        pout_misfit.axpy(kernel_value_n[self.rank][m] * kernel_value_m[self.rank][m], hessian_phat)

                        pout_kernel.axpy(kernel_gradient_m[self.rank][m].inner(phat_subs[n]),
                                         kernel_gradient_n[self.rank][m])
                        R_coeff += kernel_value_n[self.rank][m] * kernel_value_m[self.rank][m]

                    phelp = self.particle.generate_vector()
                    if not self.options["is_projection"]:
                        self.model.prior.R.mult(pout_misfit, phelp)
                    pout_misfit.zero()
                    pout_misfit.axpy(1.0, phelp)

                    pout_subs[m].axpy(1.0, pout_misfit)
                    pout_subs[m].axpy(1.0, pout_kernel)
                    pout_subs[m].axpy(R_coeff, R_phat)

                # deal with block H_mm for the particles not used in the construction map
                if m >= self.particle.number_particles:
                    R_phat = self.particle.generate_vector()
                    if self.options["is_projection"]:
                        R_phat.axpy(1.0, phat_subs[m])
                    else:
                        self.model.prior.R.mult(phat_subs[m], R_phat)
                    pout_misfit = self.particle.generate_vector()
                    pout_kernel = self.particle.generate_vector()

                    # loop over each particle used for map construction to compute the expectation
                    for p in range(self.nproc):
                        for ie in range(self.particle.number_particles):
                            hessian_phat = self.particle.generate_vector()
                            self.variation.hessian(R_phat, hessian_phat, ie, p)
                            pout_misfit.axpy(kernel_value_m[p][ie] * kernel_value_m[p][ie], hessian_phat)
                            pout_kernel.axpy(kernel_gradient_m[p][ie].inner(phat_subs[m]),
                                             kernel_gradient_m[p][ie])
                            R_coeff += kernel_value_m[p][ie] * kernel_value_m[p][ie]

                    # also use the particle to compute the expectation
                    hessian_phat = self.particle.generate_vector()
                    self.variation.hessian(R_phat, hessian_phat, m, self.rank)
                    pout_misfit.axpy(kernel_value_m[self.rank][m] * kernel_value_m[self.rank][m], hessian_phat)
                    pout_kernel.axpy(kernel_gradient_m[self.rank][m].inner(phat_subs[m]),
                                     kernel_gradient_m[self.rank][m])
                    R_coeff += kernel_value_m[self.rank][m] * kernel_value_m[self.rank][m]

                    phelp = self.particle.generate_vector()
                    if not self.options["is_projection"]:
                        self.model.prior.R.mult(pout_misfit, phelp)
                    pout_misfit.zero()
                    pout_misfit.axpy(1.0, phelp)

                    pout_subs[m].axpy(1.0, pout_misfit)
                    pout_subs[m].axpy(1.0, pout_kernel)
                    pout_subs[m].axpy(R_coeff, R_phat)

            else:
                raise NotImplementedError("choose full, diagonal, or lumped")

        self.gls.local2global(pout_subs, pout)

        self.ncalls += 1

    def assemble(self):
        # assemble Hessian matrix when is_projection is true
        p = self.nproc
        M = self.particle.number_particles_all
        N = p*M
        r = self.particle.coefficient_dimension
        hess = np.zeros((M, N, r, r))
        for m in range(M):
            kernel_value_m = self.kernel.value_set[m]
            kernel_gradient_m = self.kernel.gradient_set[m]
            for n in range(N):
                p1, n1 = n // M, n % M
                kernel_value_n = self.kernel.value_set_gather[p1][n1]
                kernel_gradient_n = self.kernel.gradient_set_gather[p1][n1]
                for k in range(N):
                    p2, n2 = k // M, k % M
                    hess[m][n] += kernel_value_m[p2][n2]*kernel_value_n[p2][n2]*\
                                  (self.variation.hessian_misfit_gather[p2][n2] + np.eye(r))
                    hess[m][n] += np.outer(kernel_gradient_n[p2][n2], kernel_gradient_m[p2][n2].get_local())
        hess_gather = np.empty([p, M, N, r, r], dtype=float)
        self.comm.Allgather(hess, hess_gather)

        hess = np.zeros((N*r, N*r))
        for m in range(N):
            p1, n1 = m // M, m % M
            for n in range(N):
                hess[m*r:(m+1)*r, n*r:(n+1)*r] = hess_gather[p1][n1][n]

        # print("kernel_value", kernel_value_m, "hess", hess, "condition", np.linalg.cond(hess))
        return hess


class NewtonCoupled:
    # solve the optimization problem by Newton method with coupled linear system
    def __init__(self, model, particle, variation, kernel, options, comm):
        self.model = model  # forward model
        self.particle = particle  # set of particles, pn = particles[m]
        self.variation = variation
        self.kernel = kernel
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()

        self.gls = GlobalLocalSwitch(self.model, self.particle, self.options, self.comm)
        self.preconditioner = PreconditionerCoupled(self.model, self.particle, self.options, self.comm)
        self.hessian = HessianCoupled(self.model, self.particle, self.variation, self.kernel, self.options, self.comm)

        self.save_kernel = options["save_kernel"]
        self.add_number = options["add_number"]
        self.add_step = options["add_step"]
        self.save_step = options["save_step"]
        self.save_number = options["save_number"]
        self.plot = options["plot"]
        self.options["is_projection"] = options["is_projection"]

        self.it = 0
        self.converged = False
        self.reason = 0

        self.gradient_norm = 0.
        self.gradient_norm_init = 0.
        self.step_norm = 0.
        self.step_norm_init = 0.
        self.relative_grad_norm = 0.
        self.relative_step_norm = 0.

        self.pg_phat = 0.
        self.tol_newton = 0.
        self.tol_cg = 0.
        self.total_cg_iter = 0
        self.final_grad_norm = 0.
        self.cost_new = 0.
        self.reg_new = 0.
        self.misfit_new = 0.

        self.cost_new_set = np.zeros(self.particle.number_particles_all)
        self.reg_new_set = np.zeros(self.particle.number_particles_all)
        self.misfit_new_set = np.zeros(self.particle.number_particles_all)

    def gradientCoupled(self, gradient):
        # compute gradient in a vector of length parameter_dimension * number_particles

        grad_subs = [self.particle.generate_vector() for n in range(self.particle.number_particles_all)]

        gradient_norm = 0
        for m in range(self.particle.number_particles_all):
            if self.save_kernel:
                kernel_value = self.kernel.value_set[m]
                kernel_gradient = self.kernel.gradient_set[m]
            else:
                kernel_value = self.kernel.values(m)
                kernel_gradient = self.kernel.gradients(m)

            if self.kernel.delta_kernel:
                gp_misfit = self.particle.generate_vector()
                gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
                grad_subs[m].axpy(kernel_value[self.rank][m], gp_misfit)
                grad_subs[m].axpy(-1.0, kernel_gradient[self.rank][m])
            else:
                for p in range(self.nproc):
                    for ie in range(self.particle.number_particles):  # take the expectation
                        gp_misfit = self.particle.generate_vector()
                        gp_misfit.set_local(self.variation.gradient_gather[p][m].get_local())
                        grad_subs[m].axpy(kernel_value[p][ie], gp_misfit)
                        grad_subs[m].axpy(-1.0, kernel_gradient[p][ie])

                # also use the particle to compute the expectation
                if m >= self.particle.number_particles:
                    gp_misfit = self.particle.generate_vector()
                    gp_misfit.set_local(self.variation.gradient_gather[self.rank][m].get_local())
                    grad_subs[m].axpy(kernel_value[self.rank][m], gp_misfit)
                    grad_subs[m].axpy(-1.0, kernel_gradient[self.rank][m])

            if self.options["is_projection"]:
                gradient_norm += grad_subs[m].inner(grad_subs[m])
            else:
                tmp = self.model.generate_vector(PARAMETER)
                self.model.prior.Msolver.solve(tmp, grad_subs[m])
                gradient_norm += grad_subs[m].inner(tmp)

        self.gls.local2global(grad_subs, gradient)

        gradient_norm = np.sqrt(gradient_norm)

        return gradient_norm

    def communication(self, phat):

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

        self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                             or self.variation.gauss_newton_approx_hold
        self.variation.update(self.particle)
        self.gls.update(self.particle)
        self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
        self.kernel.update(self.particle, self.variation)
        self.hessian.update(self.particle, self.variation, self.kernel)

        self.it = 0
        self.converged = False

        if self.save_number:
            self.kernel.save_values(self.save_number, self.it)
            self.particle.save(self.save_number, self.it)
            self.variation.save_eigenvalue(self.save_number, self.it)
            if self.plot:
                self.variation.plot_eigenvalue(self.save_number, self.it)

        while self.it < max_iter and (self.converged is False):

            if self.options["type_parameter"] is 'vector' and self.plot:
                self.particle.plot_particles(self.particle, self.it)

            phat = dl.Function(self.gls.Vh_Global).vector()

            self.pg_phat = np.zeros(self.particle.number_particles_all)
            # evaluate gradient
            gradient = dl.Function(self.gls.Vh_Global).vector()
            self.gradient_norm = self.gradientCoupled(gradient)

            if self.options["is_projection"]:
                # do communication for gradient
                if self.nproc > 1:
                    gradient_array = gradient.get_local().astype(float)
                    gradient_gather_array = np.empty(
                        [self.nproc, self.particle.dimension * self.particle.number_particles_all], dtype=float)
                    self.comm.Allgather(gradient_array, gradient_gather_array)
                    gradient_norm_tmp = np.empty([1], dtype=float)
                    gradient_norm = np.array([self.gradient_norm**2]).astype(float)
                    self.comm.Allreduce(gradient_norm, gradient_norm_tmp, op=MPI.SUM)
                    self.gradient_norm = np.sqrt(gradient_norm_tmp[0])

                    grad_array = np.zeros(self.nproc * self.particle.dimension * self.particle.number_particles_all)
                    Md = self.particle.dimension * self.particle.number_particles_all
                    for p in range(self.nproc):
                        grad_array[p*Md:(p+1)*Md] = gradient_gather_array[p]
                else:
                    grad_array = gradient.get_local()

                hess_array = self.hessian.assemble()
                phat_array = np.linalg.solve(hess_array, -grad_array)
                Md = self.particle.number_particles_all*self.particle.dimension
                phat.set_local(phat_array[self.rank*Md:(self.rank+1)*Md])

            else:
                # set tolerance for CG iteration
                self.tol_cg = min(cg_coarse_tolerance, np.sqrt(self.gradient_norm / self.gradient_norm_init))

                # set Hessian and CG solver
                solver = CGSolverSteihaug(comm=self.model.prior.R.mpi_comm())
                solver.set_operator(self.hessian)
                solver.set_preconditioner(self.preconditioner)
                solver.parameters["rel_tolerance"] = self.tol_cg
                solver.parameters["zero_initial_guess"] = True
                solver.parameters["print_level"] = print_level
                solver.parameters["max_iter"] = 20

                # solve the Newton linear system
                solver.solve(phat, -gradient)

                # # replace Hessian with its Gauss Newton approximation if CG reached a negative direction
                # if solver.reasonid == 2:
                #     self.hessian.variation.low_rank_Hessian = False
                #     self.hessian.variation.gauss_newton_approx = True
                #     solver.set_operator(self.hessian)
                #     solver.solve(phat, -gradient)

                self.total_cg_iter += self.hessian.ncalls

            # set tolerance for Newton iteration
            if self.it == 0 or self.particle.number_particles_all > self.particle.number_particles_all_old:
                self.gradient_norm_init = self.gradient_norm
                self.tol_newton = max(abs_tol, self.gradient_norm_init * rel_tol)

            phat_subs = [self.particle.generate_vector() for n in range(self.particle.number_particles_all)]
            self.gls.global2local(phat, phat_subs)

            grad_subs = [self.particle.generate_vector() for n in range(self.particle.number_particles_all)]
            self.gls.global2local(gradient, grad_subs)
            for m in range(self.particle.number_particles_all):
                self.pg_phat[m] = phat_subs[m].inner(grad_subs[m])

            phat_subs = self.communication(phat_subs)  # gather the coefficients to all processors

            # step for particle update, pstep(x) = sum_n phat_n k_n(x)
            self.pstep_norm = np.zeros(self.particle.number_particles_all)
            pstep = [self.particle.generate_vector() for m in range(self.particle.number_particles_all)]
            deltap = [self.model.generate_vector(PARAMETER) for m in range(self.particle.number_particles_all)]
            for m in range(self.particle.number_particles_all):
                for p in range(self.nproc):
                    for n in range(self.particle.number_particles):
                        pstep[m].axpy(self.kernel.value_set_gather[p][n][self.rank][m], phat_subs[p][n])
                    if m >= self.particle.number_particles:
                        pstep[m].axpy(self.kernel.value_set_gather[p][m][self.rank][m], phat_subs[p][m])

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

            self.cost_mean, self.cost_std = np.mean(self.cost_new), np.std(self.cost_new)

            # move all particles in the new directions, pm = pm + self.alpha[m] * sum_n phat[n] * k(pn, pm)
            self.particle.move(self.alpha, pstep)

            # compute the norm of the step/direction to move
            self.step_norm = self.pstep_norm * self.alpha

            self.relative_grad_norm = np.divide(self.gradient_norm, self.gradient_norm_init)
            self.relative_step_norm = np.divide(self.step_norm, self.step_norm_init)

            # print data
            if print_level >= -1:
                if self.rank == 0:
                    print("\n{0:4} {1:4} {2:4} {3:5} {4:15} {5:15} {6:15} {7:15} {8:14} {9:14} {10:14} {11:14}".format(
                         "it", "cpu", "m", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "||dm||L2", "alpha", "tolcg"))
                for m in range(self.particle.number_particles_all):
                    print("{0:3d} {1:3d} {2:3d} {3:4d} {4:15e} {5:15e} {6:15e} {7:15e} {8:14e} {9:14e} {10:14e} {11:14e}".format(
                        self.it, self.rank, m, self.hessian.ncalls, self.cost_new[m], self.misfit_new[m], self.reg_new[m],
                        self.pg_phat[m], self.relative_grad_norm, self.relative_step_norm[m], self.alpha[m], self.tol_cg))

            # verifying stopping criteria
            if self.gradient_norm <= self.tol_newton:
                self.converged = True
                self.reason = 1
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number and np.mod(self.it, self.save_step) == 0:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)
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
                if self.save_number and np.mod(self.it, self.save_step) == 0:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)
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
                if self.save_number and np.mod(self.it, self.save_step) == 0:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)
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
                if self.save_number and np.mod(self.it, self.save_step) == 0:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)
                break

            self.it += 1

            if self.it == max_iter:
                self.converged = False
                self.reason = 0
                print("Termination reason: ", self.options["termination_reasons"][self.reason])
                if self.save_number and np.mod(self.it, self.save_step) == 0:
                    self.kernel.save_values(self.save_number, self.it)
                    self.particle.save(self.save_number, self.it)
                    self.variation.save_eigenvalue(self.save_number, self.it)
                    if self.plot:
                        self.variation.plot_eigenvalue(self.save_number, self.it)
                break

            # add new particles if needed, try different adding criteria, e.g., np.max(self.tol_cg) < beta^{-t}
            if self.add_number and np.mod(self.it, self.add_step) == 0:
                self.particle.add(self.variation)
                self.gls.update(self.particle)
                self.preconditioner.update(self.particle)

                if self.particle.number_particles_all > self.particle.number_particles_all_old:
                    for m in range(self.particle.number_particles_all_old, self.particle.number_particles_all):
                        self.step_norm_init = np.insert(self.step_norm_init, m-self.particle.number_particles_add, 0.)

            # update variation, kernel, and hessian with new particles before solving the Newton linear system
            self.variation.gauss_newton_approx = (self.it < self.variation.max_iter_gauss_newton_approx) \
                                                 or self.variation.gauss_newton_approx_hold
            relative_step_norm = np.max(self.relative_step_norm)
            relative_step_norm_reduce = np.zeros(1, dtype=float)
            self.comm.Allreduce(relative_step_norm, relative_step_norm_reduce, op=MPI.MAX)
            self.relative_step_norm = relative_step_norm_reduce[0]
            self.variation.update(self.particle, self.it, self.relative_step_norm)
            self.gls.update(self.particle)
            self.kernel.delta_kernel = (self.it < self.kernel.max_iter_delta_kernel) or self.kernel.delta_kernel_hold
            self.kernel.update(self.particle, self.variation)
            self.hessian.update(self.particle, self.variation, self.kernel)

            # save the particles for visualization and plot the eigenvalues at each particle
            if self.save_number and np.mod(self.it, self.save_step) == 0:
                self.kernel.save_values(self.save_number, self.it)
                self.particle.save(self.save_number, self.it)
                self.variation.save_eigenvalue(self.save_number, self.it)
                if self.plot:
                    self.variation.plot_eigenvalue(self.save_number, self.it)
