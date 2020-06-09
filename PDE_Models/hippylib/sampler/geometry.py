#!/usr/bin/env python
"""
Class definition of Reynolds-Averaged Navier-Stokes (RANS) model
modified from https://bitbucket.org/quantum_project/quantum_apps/branch/parallel-dev
written in FEniCS 1.6.0
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
-----------------------------------
The purpose of this script is to obtain geometric quantities, log-likelihood, its gradient and the associated metric (Gauss-Newton) using adjoint methods.
--To run demo:                     python RANS.py # to compare with the finite difference method
--To initialize problem:     e.g.  rans=RANS(args)
--To obtain geometric quantities:  loglik,grad,HessApply,eigs = rans.get_geom # log-likelihood, gradient, metric action and estimated eigen-decomposition.
                                   which calls _get_misfit (_soln_fwd), _get_grad (_soln_adj), and _get_HessApply (_soln_fwd2,_soln_adj2) resp.
--To save PDE solutions:           rans.save()
                                   fwd: forward solution; adj: adjoint solution; %fwd2: 2nd order forward; adj2: 2nd order adjoint.
--To plot PDE solutions:           rans.plot()
-----------------------------------
Created July 7, 2016
Modified January 12, 2017
memo: this script works with hippyib https://bitbucket.org/hippylibdev/hippylib
Modified January 27, 2017
memo: this script can be safely run in parallel by 'mpirun'.
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__credits__ = ["Umberto Villa"]
__license__ = "GPL"
__version__ = "5.2"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; slan@caltech.edu; lanzithinking@outlook.com"

# import modules
import dolfin as dl
import numpy as np

# import sys
# sys.path.append("../")
# from utils import *

from ..modeling.variables import STATE, PARAMETER, ADJOINT
from ..modeling.posterior import GaussianLRPosterior
from ..modeling.reducedHessian import ReducedHessian
from ..algorithms.NewtonCG import ReducedSpaceNewtonCG
from ..utils.random import Random
from .whiten import *
from .randomizedEigensolver_ext import *


class Geometry:

    def __init__(self, model=None):
        """
        Initialize the model so that it becomes self-contained and ready to provide geometric quantities for Bayesian inverse problems.
        """
        self.model = model
        self.prior = model.prior
        self.whtprior = wht_prior(self.prior)
        self.x = model.generate_vector()
        self.cost = None
        self.mpi_comm = self.model.prior.R.mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        self.nproc = dl.MPI.size(self.mpi_comm)
        self.randomGen = Random(seed=1)
        # count PDE solving times
        self.soln_count = np.zeros(4)  # TODO: implement counting
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively

    # def setup(self, seed=2017, **kwargs):
    #     # set (common) random seed
    #     if self.nproc > 1:
    #         Random.split(self.rank, self.nproc, 1000000, seed)
    #     else:
    #         Random.seed(seed)
    #     np.random.seed(seed)
    #     # set FEM
    #     self._set_FEM()
    #     print('\nSpace geometry is determined.')
    #     # set pde
    #     self._set_pde(src4init=kwargs.pop('src4init', 'solution'))
    #     print('\nPhysical PDE model is defined.')
    #     # set prior
    #     self._set_prior(**kwargs)
    #     print('\nPrior models are specified.')
    #     # set misfit
    #     self._set_misfit();
    #     print('\nLikelihood model is obtained.')
    #     # 7. Set up the inverse problem model, which consits of a pde, a prior, and a misfit
    #     self.model_stat = Model(self.pde, self.prior, self.misfit)

    # def _set_FEM(self):
    #     """
    #     Define the finite element method.
    #     """
    #     # 2. Define the finite element spaces
    #     Vh_velocity = dl.VectorFunctionSpace(self.geo.mesh, 'CG', 2)
    #     Vh_pressure = dl.FunctionSpace(self.geo.mesh, 'CG', 1)
    #     Vh_k = dl.FunctionSpace(self.geo.mesh, 'CG', 1)
    #     Vh_e = dl.FunctionSpace(self.geo.mesh, 'CG', 1)
    #     self.Vh_help = dl.FunctionSpace(self.geo.mesh, "DG", 0)
    #     self.Vhs = [Vh_velocity, Vh_pressure, Vh_k, Vh_e]
    #
    #     self.Vh_STATE = dl.MixedFunctionSpace(self.Vhs)
    #
    #     self.Vh_PARAMETER = dl.FunctionSpace(self.geo.mesh, 'CG', 1)
    #
    #     self.Vh = [self.Vh_STATE, self.Vh_PARAMETER, self.Vh_STATE]
    #
    #     # 3. Define boundary conditions
    #     self._scalar_zero = dl.Expression("0.")
    #     self._vector_zero = dl.Expression(("0.", "0."))
    #     # Use the DNS at the inflow
    #     self._datafile = "../DNS_data/PlaneJet_data_all.dat"
    #     self.u_fun, self.k_fun, self.e_fun = loadDNSData(self._datafile, l_star=.1, C_mu=0.09, coflow=0.0)
    #
    #     # list of essential boundary conditions for the forward problem
    #     u_bc_inflow = dl.DirichletBC(self.Vh_STATE.sub(0), self.u_fun, self.geo.boundary_parts, self.geo.INLET)
    #     u_bc_axis = dl.DirichletBC(self.Vh_STATE.sub(0).sub(1), self._scalar_zero, self.geo.boundary_parts,
    #                                self.geo.AXIS)
    #     k_bc_inflow = dl.DirichletBC(self.Vh_STATE.sub(2), self.k_fun, self.geo.boundary_parts, self.geo.INLET)
    #     e_bc_inflow = dl.DirichletBC(self.Vh_STATE.sub(3), self.e_fun, self.geo.boundary_parts, self.geo.INLET)
    #     self.ess_bcs = [u_bc_inflow, u_bc_axis, k_bc_inflow, e_bc_inflow]
    #
    #     # list of (homogeneous) essential boundary conditions for the adjoint and incremental problems
    #     u_bc_inflow0 = dl.DirichletBC(self.Vh_STATE.sub(0), self._vector_zero, self.geo.boundary_parts, self.geo.INLET)
    #     u_bc_axis0 = dl.DirichletBC(self.Vh_STATE.sub(0).sub(1), self._scalar_zero, self.geo.boundary_parts,
    #                                 self.geo.AXIS)
    #     k_bc_inflow0 = dl.DirichletBC(self.Vh_STATE.sub(2), self._scalar_zero, self.geo.boundary_parts, self.geo.INLET)
    #     e_bc_inflow0 = dl.DirichletBC(self.Vh_STATE.sub(3), self._scalar_zero, self.geo.boundary_parts, self.geo.INLET)
    #     self.ess_bcs0 = [u_bc_inflow0, u_bc_axis0, k_bc_inflow0, e_bc_inflow0]

    # def _set_prior(self, gamma=3., delta=3.):
    #     """
    #     Define the 2d Laplacian prior with covariance
    #     C = (\delta I + \gamma \div Theta \grad) ^ {-2}.
    #
    #     The magnitude of \delta\gamma governs the variance of the samples, while
    #     the ratio \frac{\gamma}{\delta} governs the correlation length.
    #
    #     Here Theta is a s.p.d tensor that models anisotropy in the covariance kernel.
    #     """
    #     # Define the Prior
    #     Theta = dl.Constant(((15. * self.D, 0.), (0., 1. * self.D)))
    #     self.prior = BiLaplacianPrior(self.Vh_PARAMETER, gamma, delta, Theta)
    #     # Whiten the Prior
    #     self.whtprior = wht_prior(self.prior)

    # def _set_pde(self, **kwargs):
    #     """
    #     Define and initialize the physical (forward) RANS model.
    #     """
    #     # 4. Set up the RANS model
    #     # Instantiate the model (i.e. the class that manages weak forms for the forward problem)
    #     model_phys = RANSModel_inadeguate_Cmu(self.Vh_STATE, self.nu, self.geo.ds(self.geo.FARFIELD),
    #                                           self._vector_zero, self._scalar_zero, self._scalar_zero,
    #                                           self.C_mu, self.sigma_k, self.sigma_e, self.C_e1, self.C_e2)
    #
    #     # Initialize the state variable if not existing
    #     self._init_states(kwargs.pop('src4init', 'solution'))
    #
    #     # Create the adjoint/derivative capable forward problem
    #     self.pde = RANSProblem(self.Vh, model_phys, self.ess_bcs, self.ess_bcs0, self.states_fwd.vector(),
    #                            verbose=(self.rank == 0))
    #
    # def _init_states(self, src4init='map_solution'):
    #     """
    #     Initialize the states from existing solution or DNS data.
    #     """
    #     # Initialize the state variable using the DNS data or existing solution
    #     self.states_fwd = dl.Function(self.Vh_STATE)
    #     f_name = src4init + '.h5'
    #     self.from_file = os.path.isfile(f_name)
    #     if self.from_file:
    #         input_file = dl.HDF5File(self.mpi_comm, f_name, "r")
    #         input_file.read(self.states_fwd, "state")
    #         input_file.close()
    #     elif src4init is 'DNS' or not self.from_file:
    #         upke_init = [self.u_fun, self._scalar_zero, self.k_fun, self.e_fun]
    #         upkefun = [dl.interpolate(fi, Vi) for (fi, Vi) in zip(upke_init, self.Vhs)]
    #         assigner_upke = dl.FunctionAssigner(self.Vh_STATE, self.Vhs)
    #         assigner_upke.assign(self.states_fwd, upkefun)
    #         self.from_file = False
    #     else:
    #         self.from_file = False
    #         if self.rank == 0:
    #             print('States have initialized at zero. Please provide data source if you want otherwise.')

    # def _soln_fwd(self, parameter=None, fresh_init=False, para_cont=True, **kwargs):
    #     """
    #     Solve the forward equation.
    #     """
    #     if parameter is None:
    #         parameter = self.prior.mean
    #     # 5. Solve the forward problem.
    #     # Initialize the state variable if not existing
    #     if not hasattr(self, 'states_fwd') or fresh_init:
    #         self._init_states(kwargs.pop('src4init', 'solution'))
    #
    #     x = [self.pde.generate_state(), parameter]
    #     x[STATE].axpy(1., self.states_fwd.vector())
    #     self.pde.solveFwd(x[STATE], x, 1e-6)
    #     if para_cont:
    #         self.pde.initial_guess.zero()
    #         self.pde.initial_guess.axpy(1., x[STATE])
    #         self.pde.setParameterContinuationSolver(x, 1e-6)
    #     #         self.states_fwd.vector().zero()
    #     #         self.states_fwd.vector().axpy(1.,x[STATE])
    #     self.states_fwd.vector()[:] = x[STATE]
    #     self.soln_count[0] += 1
    #
    #     # save once for future use
    #     if not self.from_file:
    #         xfun = vector2Function(x[STATE], self.Vh_STATE, name="solution")
    #         output_file = dl.HDF5File(self.mpi_comm, "solution.h5", "w")
    #         output_file.write(xfun, "state")
    #         output_file.close()
    #
    # def _set_misfit(self):
    #     """
    #     Define the data-misfit function and the Statistical (inverse) model.
    #     """
    #     # 6. Set up the data misfit functional (i.e. the negative log likelihood)
    #     self.misfit = DNSDataMisfit(self.Vh_STATE, self.Vhs, self.ess_bcs0, self._datafile, self.geo.dx(self.geo.DNS))
    #
    # #         # 7. Set up the inverse problem model, which consits of a pde, a prior, and a misfit
    # #         self.model_stat = Model(self.pde,self.prior, self.misfit)

    # def _soln_adj(self, parameter=None):
    #     """
    #     Solve the adjoint equation.
    #     """
    #     if parameter is None:
    #         parameter = self.prior.mean
    #     # solve the adjoint problem
    #     x = [self.states_fwd.vector(), parameter, self.model.generate_vector(ADJOINT)]
    #
    #     self.model.solveAdj(x[ADJOINT], x, 1e-9)
    #     self.states_adj = dl.Function(self.Vh_STATE)
    #     #         self.states_adj.vector().axpy(1.,x[ADJOINT])
    #     self.states_adj.vector()[:] = x[ADJOINT]
    #     self.soln_count[1] += 1
    #
    # def _get_misfit(self, parameter=None, **kwargs):
    #     """
    #     Compute the misfit for given parameter.
    #     """
    #     if parameter is None:
    #         parameter = self.prior.mean
    #     # solve the forward problem
    #     self._soln_fwd(parameter, **kwargs)
    #     x = [self.states_fwd.vector(), parameter]
    #     misfit = self.model.misfit.cost(x)
    #     return misfit

    # def _get_grad(self, parameter=None, MF_only=True):
    #     """
    #     Compute the gradient of misfit (default), or the gradient of negative log-posterior for given parameter.
    #     """
    #     if parameter is None:
    #         parameter = self.prior.mean
    #     # 8. Compute gradient using adjoint
    #     # solve the adjoint problem
    #     self._soln_adj(parameter)
    #     x = [self.states_fwd.vector(), parameter, self.states_adj.vector()]
    #     grad = self.model.generate_vector(PARAMETER)
    #     #         if MF_only:
    #     #             # misfit gradient
    #     #             self.pde.eval_da(x,grad)
    #     #         else:
    #     #             # negative log-posterior gradient
    #     #             gradnorm = self.model.evalGradientParameter(x, grad)
    #     gradnorm = self.model.evalGradientParameter(x, grad, misfit_only=MF_only)
    #     return grad

    # def _get_HessApply(self, parameter=None, GN_appx=True, MF_only=True):
    #     """
    #     Compute the Hessian apply (action) for given parameter,
    #     default to the Gauss-Newton approximation.
    #     """
    #     if parameter is None:
    #         parameter = self.prior.mean
    #     # Compute Hessian Apply
    #     x = [self.states_fwd.vector(), parameter, self.states_adj.vector()]
    #     # set point for Hessian evaluation
    #     self.model.setPointForHessianEvaluations(x)
    #     HessApply = ReducedHessian(self.model, 1e-9, gauss_newton_approx=GN_appx, misfit_only=MF_only)
    #     return HessApply

    def get_geom(self, parameter=None, geom_ord=[0], whitened=True, log_level=dl.ERROR, **kwargs):
        """
        Get necessary geometric quantities including log-likelihood (0), adjusted gradient (1),
        Hessian apply (1.5), and its eigen-decomposition using randomized algorithm (2).
        """
        if parameter is None:
            parameter = self.prior.mean
        loglik = None
        agrad = None
        HessApply = None
        eigs = None
        # set log level: DBG(10), TRACE(13), PROGRESS(16), INFO(20,default), WARNING(30), ERROR(40), or CRITICAL(50)
        dl.set_log_level(log_level)

        # un-whiten if necessary
        if whitened:
            parameter = self.whtprior.v2u(parameter)

        self.x[PARAMETER].zero()
        self.x[PARAMETER].axpy(1., parameter)

        # get log-likelihood
        if any(s >= 0 for s in geom_ord):
            self.model.solveFwd(self.x[STATE], self.x)
            loglik = -self.model.misfit.cost(self.x)
            self.cost = self.model.cost(self.x)[0]
            self.soln_count[0] += 1

        # get gradient
        if any(s >= 1 for s in geom_ord):
            grad = self.model.generate_vector(PARAMETER)
            self.model.solveAdj(self.x[ADJOINT], self.x)
            self.soln_count[1] += 1
            self.model.evalGradientParameter(self.x, grad, misfit_only=kwargs.pop('MF_only', True))  # check misfit
            agrad = self.model.generate_vector(PARAMETER)
            agrad.axpy(-1., grad)

            if whitened:
                agrad_ = agrad.copy()
                agrad.zero()
                self.whtprior.C_act(agrad_, agrad, comp=0.5, transp=True)

        # get Hessian Apply
        if any(s >= 1.5 for s in geom_ord):
            self.model.setPointForHessianEvaluations(self.x)
            HessApply = ReducedHessian(self.model, 1e-9, misfit_only=kwargs.pop('MF_only', True))

            if whitened:
                HessApply = wht_Hessian(self.whtprior, HessApply)
            if np.max(geom_ord) <= 1.5:
                # adjust the gradient
                Hu = self.model.generate_vector(PARAMETER)
                HessApply.mult(parameter, Hu)
                agrad.axpy(1., Hu)
                if not kwargs.pop('MF_only', True):
                    Ru = self.model.generate_vector(PARAMETER)
                    #                     self.prior.R.mult(parameter,Ru)
                    self.prior.grad(parameter, Ru)
                    agrad.axpy(-1., Ru)

        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s > 1 for s in geom_ord):
            k = kwargs['k'] if 'k' in kwargs else kwargs['incr_k'] if 'incr_k' in kwargs else 20
            p = kwargs['p'] if 'p' in kwargs else 10
            if len(kwargs) == 0:
                kwargs['k'] = k
            #             if self.rank == 0:
            #                 print('Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.'.format(k,p))
            Omega = MultiVector(parameter, k + p)
            self.randomGen.normal(1., Omega)
            # for i in xrange(k + p):
            #     Random.normal(Omega[i], 1., True)
            if whitened:
                eigs = singlePassGx(HessApply, self.prior.M, self.prior.Msolver, Omega, **kwargs)
            else:
                eigs = doublePassG(HessApply, self.prior.R, self.prior.Rsolver, Omega, **kwargs)
            if any(s > 1.5 for s in geom_ord):
                # adjust the gradient using low-rank approximation
                self.post_Ga = GaussianLRPosterior(getattr(self, {True: 'wht', False: ''}[whitened] + 'prior'), eigs[0],
                                                   eigs[1])
                Hu = self.model.generate_vector(PARAMETER)
                self.post_Ga.Hlr.mult(parameter, Hu)  # post_Ga.Hlr=posterior precision
                agrad.axpy(1., Hu)
                Ru = self.model.generate_vector(PARAMETER)
                getattr(self, {True: 'wht', False: ''}[whitened] + 'prior').grad(parameter, Ru)
                agrad.axpy(-1., Ru)

        return loglik, agrad, HessApply, eigs

    # def get_eigs(self, parameter=None, whitened=False, **kwargs):
    #     """
    #     Get the eigen-decomposition of Hessian action directly using randomized algorithm.
    #     """
    #     if parameter is None:
    #         parameter = self.prior.mean
    #
    #     # un-whiten if necessary
    #     if whitened:
    #         parameter = self.whtprior.v2u(parameter)
    #
    #     # solve the forward problem
    #     self._soln_fwd(parameter, fresh_init=kwargs.pop('fresh_init', False),
    #                    src4init=kwargs.pop('src4init', 'map_solution'), para_cont=kwargs.pop('para_cont', True))
    #     # solve the adjoint problem
    #     self._soln_adj(parameter)
    #     # get Hessian Apply
    #     HessApply = self._get_HessApply(parameter, kwargs.pop('GN_appx', True),
    #                                     kwargs.pop('MF_only', True))  # Hmisfit if MF is true
    #     if whitened:
    #         HessApply = wht_Hessian(self.whtprior, HessApply)
    #     # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
    #     k = kwargs['k'] if 'k' in kwargs else kwargs['incr_k'] if 'incr_k' in kwargs else 20
    #     p = kwargs['p'] if 'p' in kwargs else 10
    #     if len(kwargs) == 0:
    #         kwargs['k'] = k
    #     #         if self.rank == 0:
    #     #             print('Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.'.format(k,p))
    #     Omega = MultiVector(parameter, k + p)
    #     for i in xrange(k + p):
    #         Random.normal(Omega[i], 1., True)
    #     if whitened:
    #         eigs = singlePassGx(HessApply, self.prior.M, self.prior.Msolver, Omega, **kwargs)
    #     else:
    #         eigs = doublePassG(HessApply, self.prior.R, self.prior.Rsolver, Omega, **kwargs)
    #
    #     return eigs

    def get_MAP(self, rand_init=False, SAVE=True):
        """
        Get the maximum a posterior (MAP).
        """
        import time
        if self.rank == 0:
            print('\n\nObtaining the maximum a posterior (MAP)...')
        # setup Newton solver
        m0 = self.prior.mean.copy()
        if rand_init:
            noise = dl.Vector()
            self.prior.init_vector(noise, "noise")
            Random.normal(noise, 1., True)
            self.prior.sample(noise, m0)
        solver = ReducedSpaceNewtonCG(self.model)
        solver.parameters["rel_tolerance"] = 1e-9
        solver.parameters["abs_tolerance"] = 1e-12
        solver.parameters["max_iter"] = 25
        solver.parameters["inner_rel_tolerance"] = 1e-9
        solver.parameters["c_armijo"] = 1e-4
        solver.parameters["GN_iter"] = 5
        if self.rank == 0:
            solver.parameters["print_level"] = 0
        else:
            solver.parameters["print_level"] = -1
        # # initialize the states
        # self._init_states(src4init='solution')
        # x = [self.states_fwd.vector(), m0]
        # self.pde.setParameterContinuationSolver(x, 1e-6)
        # # solve for MAP
        start = time.time()
        MAP = solver.solve(m0)
        end = time.time()
        if self.rank == 0:
            print('\nTime used is %.4f' % (end - start))

        parameter = vector2Function(MAP[PARAMETER], self.model.problem.Vh[PARAMETER])

        # if SAVE:
        #     self._check_folder()
        #     MAP_file = dl.HDF5File(self.mpi_comm, os.path.join(self.savepath, "map_solution.h5"), "w")
        #     state = vector2Function(MAP[STATE], self.Vh[STATE])
        #     u_map, p_map, k_map, e_map = state.split(deepcopy=True)
        #     MAP_file.write(state, "state")
        #     parameter = vector2Function(MAP[PARAMETER], self.Vh[PARAMETER])
        #     MAP_file.write(parameter, "parameter")
        #     MAP_file.write(vector2Function(MAP[ADJOINT], self.Vh[ADJOINT]), "adjoint")
        #     MAP_file.write(vector2Function(self.model.misfit.d, self.Vh[STATE]), "data")
        #     MAP_file.write(vector2Function(MAP[STATE] - self.model.misfit.d, self.Vh[STATE]), "misfit")
        #     MAP_file.write(dl.project(self.pde.model.nu_t(k_map, e_map, parameter), self.Vh_help), "nu_t")
        #     MAP_file.close()

        if self.rank == 0:
            if solver.converged:
                print('\nConverged in ', solver.it, ' iterations.')
            else:
                print('\nNot Converged')

            print('Termination reason: ', solver.termination_reasons[solver.reason])
            print('Final gradient norm: ', solver.final_grad_norm)
            print('Final cost: ', solver.final_cost)

        return parameter

#     def _check_folder(self, fld_name='result'):
#         """
#         Check the existence of folder for storing result and create one if not
#         """
#         import os
#         if not hasattr(self, 'savepath'):
#             cwd = os.getcwd()
#             self.savepath = os.path.join(cwd, fld_name)
#         if not os.path.exists(self.savepath):
#             print('Save path does not exist; created one.')
#             os.makedirs(self.savepath)
#
#     def save_soln(self, sep=False):
#         """
#         Save the solutions to file.
#         """
#         # name settings
#         self.sols = ['fwd', 'adj']
#         self.sols_names = ['forward', 'adjoint']
#         self.sub_titles = ['u', 'p', 'k', 'e']
#         self._check_folder()
#         for i, sol in enumerate(self.sols):
#             # get solution
#             sol_name = '_'.join(['states', sol])
#             try:
#                 soln = getattr(self, sol_name)
#             except AttributeError:
#                 print(self.sols_names[i] + 'solution not found!')
#                 pass
#             else:
#                 if not sep:
#                     dl.File(os.path.join(self.savepath, sol_name + '.xdmf')) << soln
#                 else:
#                     soln = soln.split(True)
#                     for j, splt in enumerate(self.sub_titles):
#                         dl.File(os.path.join(self.savepath, '_'.join([self.sols_names[i], splt]) + '.pvd')) << soln[j]
#
#     def _plot_vtk(self, SAVE=False):
#         for i, sol in enumerate(self.sols):
#             # get solution
#             try:
#                 soln = getattr(self, '_'.join(['states', sol]))
#             except AttributeError:
#                 print(self.sols_names[i] + 'solution not found!')
#                 pass
#             else:
#                 soln = soln.split(True)
#                 for j, splt in enumerate(self.sub_titles):
#                     fig = dl.plot(soln[j], title=self.titles[i] + ' ' + splt, rescale=True)
#                     if SAVE:
#                         fig.write_png(os.path.join(self.savepath, '_'.join([self.sols_names[i], splt]) + '.png'))
#
#     def _plot_mpl(self, SAVE=False):
#         import matplotlib.pyplot as plt
#         from util import matplot4dolfin
#         matplot = matplot4dolfin()
#         # codes for plotting solutions
#         import matplotlib as mpl
#         for i, titl in enumerate(self.titles):
#             # get solution
#             try:
#                 soln = getattr(self, '_'.join(['states', self.sols[i]]))
#             except AttributeError:
#                 print(self.sols_names[i] + 'solution not found!')
#                 pass
#             else:
#                 soln = soln.split(True)
#                 fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, num=i, figsize=(10, 6))
#                 for j, ax in enumerate(axes.flat):
#                     plt.axes(ax)
#                     sub_fig = matplot.plot(soln[j])
#                     plt.axis('tight')
#                     ax.set_title(self.sub_titles[j])
#                 cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
#                 plt.colorbar(sub_fig, cax=cax, **kw)  # TODO: fix the issue of common color range
#                 # set common titles
#                 fig.suptitle(titl)
#                 if SAVE:
#                     matplot.save(self.savepath, self.sols_names[i] + '.png', bbox_inches='tight')
#
#     def plot_soln(self, backend='matplotlib', SAVE=False):
#         """
#         Plot the solutions.
#         """
#         # title settings
#         if not hasattr(self, 'titles'):
#             self.titles = ['Forward Solution', 'Adjoint Solution']
#         if not hasattr(self, 'sols'):
#             self.sols = ['fwd', 'adj']
#         if not hasattr(self, 'sols_names'):
#             self.sols_names = ['forward', 'adjoint']
#         if not hasattr(self, 'sub_titles'):
#             self.sub_titles = ['u', 'p', 'k', 'e']
#         if SAVE:
#             self._check_folder()
#         if backend is 'matplotlib':
#             import matplotlib.pyplot as plt
#             self._plot_mpl(SAVE=SAVE)
#             plt.show()
#         elif backend is 'vtk':
#             self._plot_vtk(SAVE=SAVE)
#             interactive()
#         else:
#             raise Exception(backend + 'not found!')
#
#     def test(self, SAVE=False, PLOT=False, chk_fd=False, h=1e-4):
#         """
#         Demo to check results with the adjoint method against the finite difference method.
#         """
#         # random sample parameter
#         noise = dl.Vector()
#         self.prior.init_vector(noise, "noise")
#         #         Random.normal(noise, 1., True)
#         parameter = self.model.generate_vector(PARAMETER)
#         #         self.prior.sample(noise,parameter)
#
#         import time
#         # obtain the geometric quantities
#         if self.rank == 0:
#             print('\n\nObtaining geometric quantities with Adjoint method...')
#         start = time.time()
#         loglik, grad, _, _ = self.get_geom(parameter, geom_ord=[0, 1], whitened=False, src4init='solution',
#                                            fresh_init=False)
#         HessApply = self._get_HessApply(parameter, GN_appx=False, MF_only=False)
#         end = time.time()
#         if self.rank == 0:
#             print('Time used is %.4f' % (end - start))
#
#         # save solutions to file
#         if SAVE:
#             self.save_soln()
#         # plot solutions
#         if PLOT:
#             if self.nproc == 1:
#                 self.plot_soln(SAVE=SAVE)
#             else:
#                 print('Plotting does not work correctly in parallel!')
#                 pass
#
#         if chk_fd:
#             # check with finite difference
#             if self.rank == 0:
#                 print('\n\nTesting against Finite Difference method...')
#             start = time.time()
#             # random direction
#             Random.normal(noise, 1., True)
#             v = self.model.generate_vector(PARAMETER)
#             self.prior.sample(noise, v)
#             ## gradient
#             if self.rank == 0:
#                 print('\nChecking gradient:')
#             parameter_p = parameter.copy()
#             parameter_p.axpy(h, v)
#             loglik_p = -self._get_misfit(parameter_p)
#             parameter_m = parameter.copy()
#             parameter_m.axpy(-h, v)
#             loglik_m = -self._get_misfit(parameter_m)
#             dloglikv_fd = (loglik_p - loglik_m) / (2 * h)
#             dloglikv = grad.inner(v)
#             #             rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
#             rdiff_gradv = np.abs(dloglikv_fd - dloglikv) / v.norm('l2')
#             if self.rank == 0:
#                 print(
#                     'Relative difference of gradients in a random direction between adjoint and finite difference: %.10f' % rdiff_gradv)
#
#             # random direction
#             Random.normal(noise, 1., True)
#             w = self.model.generate_vector(PARAMETER)
#             self.prior.sample(noise, w)
#             ## metric-action
#             if self.rank == 0:
#                 print('\nChecking Metric-action:')
#             parameter_p = parameter.copy()
#             parameter_p.axpy(h, w)
#             self._soln_fwd(parameter_p)
#             gradv_p = self._get_grad(parameter_p, MF_only=False).inner(v)
#             parameter_m = parameter.copy()
#             parameter_m.axpy(-h, w)
#             self._soln_fwd(parameter_m)
#             gradv_m = self._get_grad(parameter_m, MF_only=False).inner(v)
#             dgradvw_fd = (gradv_p - gradv_m) / (2 * h)
#             dgradvw = HessApply.inner(v, w)
#             #             rdiff_Hessvw = np.abs(dgradvw_fd-dgradvw)/np.linalg.norm(v)/np.linalg.norm(w)
#             rdiff_Hessvw = np.abs(dgradvw_fd - dgradvw) / v.norm('l2') / w.norm('l2')
#             end = time.time()
#             if self.rank == 0:
#                 print(
#                     'Relative difference of Hessians in two random directions between adjoint and finite difference: %.10f' % rdiff_Hessvw)
#                 print('Time used is %.4f' % (end - start))
#
#
# if __name__ == '__main__':
#     seed = 2017
#     np.random.seed(seed)
#     rans = RANS(nx=40, ny=80)
#     rans.setup(seed)
#     #     rans.test(SAVE=True,PLOT=False,chk_fd=True,h=1e-4)
#     map_f = rans.get_MAP(rand_init=False)
#     #     f=dl.HDF5File(rans.mpi_comm,'./result/map_solution.h5',"r")
#     #     map_f=dl.Function(rans.Vh[PARAMETER],name='parameter')
#     #     f.read(map_f,'parameter')
#     #     f.close()
#     import matplotlib.pyplot as plt
#     from util import matplot4dolfin
#
#     matplot = matplot4dolfin()
#     fig = matplot.plot(map_f)
#     plt.colorbar(fig)
#     matplot.save(savepath='./result', filename='map_solution.png', bbox_inches='tight')
#     matplot.show()