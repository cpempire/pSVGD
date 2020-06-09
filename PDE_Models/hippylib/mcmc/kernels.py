# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from __future__ import absolute_import, division, print_function

from ..modeling.variables import PARAMETER, STATE
from ..utils.random import parRandom

import math
import numpy as np
import dolfin as dl


class MALAKernel:
    def __init__(self, model):
        self.model = model
        self.pr_mean = model.prior.mean
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["delta_t"]               = 0.25*1e-4
        
        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")
        
    def name(self):
        return "inf-MALA"
        
    def derivativeInfo(self):
        return 1
    
    def init_sample(self, s):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(s.u, [s.u,s.m,s.p], inner_tol)
        s.cost = self.model.cost([s.u,s.m,s.p])[2]
        self.model.solveAdj(s.p, [s.u,s.m,s.p], inner_tol)
        self.model.evalGradientParameter([s.u,s.m,s.p], s.g, misfit_only=True)
        self.model.prior.Rsolver.solve(s.Cg, s.g)
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        rho_mp = self.acceptance_ratio(current, proposed)
        rho_pm = self.acceptance_ratio(proposed, current)
        al = rho_mp - rho_pm
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        delta_t = self.parameters["delta_t"]
        parRandom.normal(1., self.noise)
        w = dl.Vector()
        self.model.prior.init_vector(w, 0)
        self.model.prior.sample(self.noise,w, add_mean=False)
        delta_tp2 = 2 + delta_t
        d_gam = self.pr_mean + (2-delta_t)/(2+delta_t) * (current.m -self.pr_mean) - (2*delta_t)/(delta_tp2)*current.Cg + math.sqrt(8*delta_t)/delta_tp2 * w
        return d_gam

    def acceptance_ratio(self, origin, destination):
        delta_t = self.parameters["delta_t"]
        m_m = destination.m - origin.m
        p_m = destination.m + origin.m - 2.*self.pr_mean
        temp = origin.Cg.inner(origin.g)
        rho_uv = origin.cost + 0.5*origin.g.inner(m_m) + \
                0.25*delta_t*origin.g.inner(p_m) + \
                0.25*delta_t*temp
        return rho_uv
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()
        
        

class pCNKernel:
    def __init__(self, model):
        self.model = model
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["s"]                     = 0.1
        
        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")
        
    def name(self):
        return "pCN"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u,current.m,None], inner_tol)
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = -proposed.cost + current.cost
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w = dl.Vector()
        self.model.prior.init_vector(w, 0)
        self.model.prior.sample(self.noise,w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w *= s
        w.axpy(1., self.model.prior.mean)
        w.axpy(np.sqrt(1. - s*s), current.m - self.model.prior.mean)
        
        return w
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand() 
    
class gpCNKernel:
    """
    Reference:
        `F. J. PINSKI, G. SIMPOSN, A. STUART, H. WEBER,
        Algorithms for Kullback-Leibler Approximation of Probability Measures in Infinite Dimensions,
        http://arxiv.org/pdf/1408.1920v1.pdf, Alg. 5.2`
    """
    def __init__(self, model, nu):
        self.model = model
        self.nu = nu
        self.prior = model.prior
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["s"]                     = 0.1
        
        self.noise = dl.Vector()
        self.nu.init_vector(self.noise, "noise")
        
    def name(self):
        return "gpCN"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u,current.m,None], inner_tol)
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0
        
    def delta(self,sample):
        dm_nu = sample.m - self.nu.mean
        dm_pr = sample.m - self.prior.mean
        return sample.cost + .5*self.prior.R.inner(dm_pr, dm_pr) - .5*self.nu.Hlr.inner(dm_nu, dm_nu)
        

    def proposal(self, current):
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = dl.Vector()
        self.nu.init_vector(w_prior, 0)
        w = dl.Vector()
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w *= s
        w.axpy(1., self.nu.mean)
        w.axpy(np.sqrt(1. - s*s), current.m - self.nu.mean)
        
        return w
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand() 
    
    
class ISKernel:
    def __init__(self, model, nu):
        self.model = model
        self.nu = nu
        self.prior = model.prior
        self.parameters = {}
        self.parameters["inner_rel_tolerance"]   = 1e-9
        
        self.noise = dl.Vector()
        self.nu.init_vector(self.noise, "noise")
        
    def name(self):
        return "IS"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u,current.m,None], inner_tol)
        current.cost = self.model.cost([current.u,current.m,None])[2]
        
    def sample(self, current, proposed): 
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if(al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0
        
    def delta(self,sample):
        dm_nu = sample.m - self.nu.mean
        dm_pr = sample.m - self.prior.mean
        return sample.cost + .5*self.prior.R.inner(dm_pr, dm_pr) - .5*self.nu.Hlr.inner(dm_nu, dm_nu)
        

    def proposal(self, current):
        #Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = dl.Vector()
        self.nu.init_vector(w_prior, 0)
        w = dl.Vector()
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=True)
        
        return w
    
    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()


class SNmapKernel:
    """
    Stochastic Newton with MAP Hessian
    """

    def __init__(self, model, nu):
        """
        Parameters:

            - :code:`model` - an object of type :code:`Model`
            - :code:`nu` - an object of type :code:`GaussianLRPosterior`
        """
        self.model = model
        self.nu = nu
        self.parameters = {}
        self.parameters["inner_rel_tolerance"] = 1e-9

        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")

        self.discard = self.model.generate_vector(PARAMETER)
        self.w = self.model.generate_vector(PARAMETER)

    def name(self):
        return "StochasticNewton_MAP"

    def derivativeInfo(self):
        return 1

    def init_sample(self, s):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(s.u, [s.u, s.m, s.p], inner_tol)
        s.cost = self.model.cost([s.u, s.m, s.p])[0]
        self.model.solveAdj(s.p, [s.u, s.m, s.p], inner_tol)
        self.model.evalGradientParameter([s.u, s.m, s.p], s.g, misfit_only=False)
        self.nu.Hlr.solve(s.Cg, s.g)

    def sample(self, current, proposed):
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        c2p = self.neg_log_rho(current, proposed)
        p2c = self.neg_log_rho(proposed, current)
        al = c2p - p2c
        if (al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def proposal(self, current):
        parRandom.normal(1., self.noise)
        self.nu.sample(self.noise, self.discard, self.w, add_mean=False)
        return current.m - current.Cg + self.w

    def neg_log_rho(self, origin, destination):
        w = destination.m - origin.m + origin.Cg
        return origin.cost + 0.5 * self.nu.Hlr.inner(w, w)

    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()


##### using local Hessian instead of Hessian at MAP

from ..modeling.posterior import GaussianLRPosterior
from ..modeling.reducedHessian import ReducedHessian
from ..algorithms.randomizedEigensolver import doublePassG
from ..algorithms.multivector import MultiVector


class lpCNKernel:
    """
    Reference:
        using local Hessian at each sample instead of at MAP in the following paper
        this does not work well yet
        `F. J. PINSKI, G. SIMPOSN, A. STUART, H. WEBER,
        Algorithms for Kullback-Leibler Approximation of Probability Measures in Infinite Dimensions,
        http://arxiv.org/pdf/1408.1920v1.pdf, Alg. 5.2`
    """

    def __init__(self, model):
        self.model = model
        self.prior = model.prior
        self.parameters = {}
        self.parameters["inner_rel_tolerance"] = 1e-9
        self.parameters["s"] = 0.1

        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")
        self.x = self.model.generate_vector()

        self.nu = None

    def name(self):
        return "lpCN"

    def derivativeInfo(self):
        return 0

    def init_sample(self, current):
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(current.u, [current.u, current.m, None], inner_tol)
        current.cost = self.model.cost([current.u, current.m, None])[0]

    def update_nu(self, sample):

        k = 50
        p = 20
        self.x[PARAMETER].zero()
        self.x[PARAMETER].axpy(1., sample)
        self.model.solveFwd(self.x[STATE], self.x)
        self.model.setPointForHessianEvaluations(self.x, gauss_newton_approx=True)
        Hmisfit = ReducedHessian(self.model, misfit_only=True)
        Omega = MultiVector(sample, k + p)
        parRandom.normal(1., Omega)

        d, U = doublePassG(Hmisfit, self.model.prior.R, self.model.prior.Rsolver, Omega, k, s=2, check=False)
        d[d < 0.] = 0.
        self.nu = GaussianLRPosterior(self.model.prior, d, U)
        self.nu.mean = sample

    def sample(self, current, proposed):
        # update nu
        self.update_nu(current.m)
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        al = self.delta(current) - self.delta(proposed)
        if (al > math.log(np.random.rand())):
            current.assign(proposed)
            return 1
        else:
            return 0

    def delta(self, sample):
        dm_nu = sample.m - self.nu.mean
        dm_pr = sample.m - self.prior.mean
        return sample.cost + .5 * self.prior.R.inner(dm_pr, dm_pr) - .5 * self.nu.Hlr.inner(dm_nu, dm_nu)

    def proposal(self, current):
        # Generate sample from the prior
        parRandom.normal(1., self.noise)
        w_prior = dl.Vector()
        self.nu.init_vector(w_prior, 0)
        w = dl.Vector()
        self.nu.init_vector(w, 0)
        self.nu.sample(self.noise, w_prior, w, add_mean=False)
        # do pCN linear combination with current sample
        s = self.parameters["s"]
        w *= s
        w.axpy(1., self.nu.mean)
        w.axpy(np.sqrt(1. - s * s), current.m - self.nu.mean)

        return w

    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()


class SNKernel:
    """
    Stochastic Newton with local Hessian
    """

    def __init__(self, model):
        """
        Parameters:

            - :code:`model` - an object of type :code:`Model`
            - :code:`nu` - an object of type :code:`GaussianLRPosterior`
        """
        self.model = model
        self.parameters = {}
        self.parameters["inner_rel_tolerance"] = 1e-9

        self.noise = dl.Vector()
        self.model.prior.init_vector(self.noise, "noise")

        self.discard = self.model.generate_vector(PARAMETER)
        self.w = self.model.generate_vector(PARAMETER)
        self.x = self.model.generate_vector()
        self.change = True

        self.nu = None

    def update_nu(self, sample):

        k = 50
        p = 20
        self.x[PARAMETER].zero()
        self.x[PARAMETER].axpy(1., sample)
        self.model.solveFwd(self.x[STATE], self.x)
        self.model.setPointForHessianEvaluations(self.x, gauss_newton_approx=True)
        Hmisfit = ReducedHessian(self.model, misfit_only=True)
        Omega = MultiVector(sample, k + p)
        parRandom.normal(1., Omega)

        d, U = doublePassG(Hmisfit, self.model.prior.R, self.model.prior.Rsolver, Omega, k, s=2, check=False)
        d[d < 0.] = 0.
        self.nu = GaussianLRPosterior(self.model.prior, d, U)
        self.nu.mean = sample

    def name(self):
        return "StochasticNewton"

    def derivativeInfo(self):
        return 1

    def init_sample(self, s):
        if True: # self.nu is None:
        # if self.nu is None:
            # update nu
            self.update_nu(s.m)
        inner_tol = self.parameters["inner_rel_tolerance"]
        self.model.solveFwd(s.u, [s.u, s.m, s.p], inner_tol)
        s.cost = self.model.cost([s.u, s.m, s.p])[0]
        # print("s.cost", s.cost)
        self.model.solveAdj(s.p, [s.u, s.m, s.p], inner_tol)
        self.model.evalGradientParameter([s.u, s.m, s.p], s.g, misfit_only=False)
        self.nu.Hlr.solve(s.Cg, s.g)

    def sample(self, current, proposed):
        if True: # self.change:
            # update nu
            self.update_nu(current.m)
        proposed.m = self.proposal(current)
        self.init_sample(proposed)
        c2p = self.neg_log_rho(current, proposed)
        p2c = self.neg_log_rho(proposed, current)
        al = c2p - p2c
        if (al > math.log(np.random.rand())):
            current.assign(proposed)
            self.change = True
            return 1
        else:
            self.change = False
            return 0

    def proposal(self, current):
        parRandom.normal(1., self.noise)
        self.nu.sample(self.noise, self.discard, self.w, add_mean=False)
        return current.m - current.Cg + self.w

    def neg_log_rho(self, origin, destination):
        w = destination.m - origin.m + origin.Cg
        # return origin.cost + 0.5 * self.nu.Hlr.inner(w, w) - 0.5*np.sum(np.log(self.nu.d+1))
        return origin.cost + 0.5 * self.nu.Hlr.inner(w, w)

    def consume_random(self):
        parRandom.normal(1., self.noise)
        np.random.rand()
