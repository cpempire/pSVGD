#!/usr/bin/env python
"""
Class definition of whitening Gaussian prior
Given N(mu,C), output N(0,I)
---------------------------------------------------------------
written in FEniCS 2017.1.0-dev, with backward support for 1.6.0
Shiwei Lan @ Caltech, Sept. 2017
--------------------------------
Created March 28, 2017
----------------------
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017, The EQUiPS project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@outlook.com"

import dolfin as dl
import numpy as np

class wht_prior:
    """
    Whiten the Gaussian prior N(mu,C), assembled(C^(-1)) = R = A * M^(-1) * A
    """
    def __init__(self,prior):
        self.prior=prior
        self.R=prior.M # for the convenience of defining GaussianLRPosterior
        self.Rsolver=prior.Msolver
    
    def init_vector(self,x,dim):
        """
        Inizialize a vector x to be compatible with the range/domain of M.
        If dim == "noise" initialize x to be compatible with the size of white noise used for sampling.
        """
        if dim == "noise":
            self.prior.sqrtM.init_vector(x,1)
        else:
            self.prior.init_vector(x,dim)
    
    def generate_vector(self,dim=0,v=None):
        """
        Generate/initialize a dolfin generic vector to be compatible with the size of dof.
        """
        vec = dl.Vector()
        self.init_vector(vec,dim)
        if v is not None:
            vec[:]=v
        return vec
    
    def cost(self,x):
        """
        The whitened prior potential
        """
        Mx = self.generate_vector()
        self.prior.M.mult(x,Mx)
        return .5*Mx.inner(x)
    
    def grad(self,x,out):
        """
        The gradient of whitened prior potential
        """
        out.zero()
        self.prior.M.mult(x,out)
    
    def sample(self,noise,s):
        """
        Sample a random function v(x) ~ N(0,I)
        vector v ~ N(0,M^(-1))
        """
        rhs=self.prior.sqrtM*noise
        self.prior.Msolver.solve(s,rhs)
    
    def C_act(self,u_actedon,Cu,comp=1,transp=False):
        """
        Calculate operation of C^comp on vector a: a --> C^comp * a
        """
        if comp==0:
            Cu[:]=u_actedon
        else:
            if comp in [0.5,1]:
                solver=getattr(self.prior,{0.5:'A',1:'R'}[comp]+'solver')
                if not transp:
                    Mu=self.generate_vector()
                    self.prior.M.mult(u_actedon,Mu)
                    solver.solve(Cu,Mu)
                else:
                    inv_u=self.generate_vector(dim=1)
                    solver.solve(inv_u,u_actedon)
                    self.prior.M.mult(inv_u,Cu)
            elif comp in [-0.5,-1]:
                multer=getattr(self.prior,{-0.5:'A',-1:'R'}[comp])
                if not transp:
                    _u=self.generate_vector()
                    multer.mult(u_actedon,_u)
                    self.prior.Msolver.solve(Cu,_u)
                else:
                    invMu=self.generate_vector(dim=1)
                    self.prior.Msolver.solve(invMu,u_actedon)
                    multer.mult(invMu,Cu)
            else:
                warnings.warn('Action not defined!')
                pass
    
    def u2v(self,u,u_ref=None):
        """
        Transform the original parameter u to the whitened parameter v:=_C^(-1/2)(u-u_ref)
        """
        if u_ref is None:
            u_ref=self.prior.mean
        u_=u.copy()
        u_.axpy(-1.,u_ref)
        v=self.generate_vector(dim=1)
        self.C_act(u_,v,comp=-0.5)
        return v
    
    def v2u(self,v,u_ref=None):
        """
        Transform the whitened parameter v back to the original parameter u=_C^(1/2)v+u_ref
        """
        if u_ref is None:
            u_ref=self.prior.mean
        u=self.generate_vector(dim=1)
        self.C_act(v,u,comp=0.5)
        u.axpy(1.,u_ref)
        return u

class wht_Hessian:
    """
    Whiten the given Hessian H(u) to be H(v) := _C^(1/2) * H(u) * _C^(1/2)
    """
    def __init__(self,whtprior,HessApply):
        self.whtprior=whtprior
        self.HessApply=HessApply
    
    def mult(self,x,y):
        rtCx = self.whtprior.generate_vector(dim=1)
        self.whtprior.C_act(x,rtCx,comp=0.5)
        HrtCx = self.whtprior.generate_vector()
        self.HessApply.mult(rtCx,HrtCx)
        self.whtprior.C_act(HrtCx,y,comp=0.5,transp=True)
    
    def inner(self,x,y):
        Hy = self.whtprior.generate_vector()
        Hy.zero()
        self.mult(y,Hy)
        return x.inner(Hy)

if __name__ == '__main__':
    seed=2017
    np.random.seed(seed)
    import sys
    sys.path.append( "../../" )
    from hippylib import *
    from RANS import RANS
    rans = RANS()
    rans.setup(seed)
    whtprior = whtprioror(rans.prior)
    
    noise = dl.Vector()
    rans.prior.init_vector(noise,"noise")
    Random.normal(noise, 1., True)
    
    PARAMETER=1
    u = rans.model_stat.generate_vector(PARAMETER)
    rans.prior.sample(noise,u)
    cost_u=rans.prior.cost(u)
    grad_u=rans.model_stat.generate_vector(PARAMETER)
    rans.prior.grad(u,grad_u)
    print('The prior potential at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_u,grad_u.norm('l2')))
    dl.plot(vector2Function(u,rans.pde.Vh[PARAMETER]))
    
    v = whtprior.u2v(u)
    cost_v=whtprior.cost(v)
    grad_v=rans.model_stat.generate_vector(PARAMETER)
    whtprior.grad(v,grad_v)
    print('The potential of whitened prior at v is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_v,grad_v.norm('l2')))
    dl.plot(vector2Function(v,rans.pde.Vh[PARAMETER]))
    
    Random.normal(noise, 1., True)
    whtprior.sample(noise,v)
    cost_v=whtprior.cost(v)
    whtprior.grad(v,grad_v)
    print('The potential of whitened prior at v is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_v,grad_v.norm('l2')))
    dl.plot(vector2Function(v,rans.pde.Vh[PARAMETER]))
    
    u = whtprior.v2u(v)
    cost_u=rans.prior.cost(u)
    rans.prior.grad(u,grad_u)
    print('The prior potential at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_u,grad_u.norm('l2')))
    dl.plot(vector2Function(u,rans.pde.Vh[PARAMETER]))
    
    dl.interactive()