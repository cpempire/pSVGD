#!/usr/bin/env python
"""
Extension to randomizedEigensolver included in hIPPYlib
---------------------------------------------------------------
written in FEniCS 2017.1.0-dev, with backward support for 1.6.0
Shiwei Lan @ Caltech, Sept. 2017
--------------------------------
Created March 29, 2017
----------------------
referred to hIPPYlib https://hippylib.github.io
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017, The EQUiPS project"
__credits__ = "Umberto Villa"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@outlook.com"

import dolfin as dl
import numpy as np

# import sys
# sys.path.append( "../../" )
# from hippylib import *

from ..algorithms.multivector import MultiVector, MatMvMult, MvDSmatMult
from ..algorithms.linalg import Solver2Operator
from ..algorithms.randomizedEigensolver import singlePassG

def _lr_eig(Omega,T,k,Q):
    """
    The low-rank eigen-decomposition based on projected matrix T
    The common core component of randomized eigensolver
    """
    nvec  = Omega.nvec()
    
    assert(nvec >= k )
                  
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    
    return d, U

def singlePassG_prec(A,B,Binv,Omega,incr_k=20,s=1,check=False,dim=None,thld=.01):
    """
    Get partial generalized eigen-pairs of pencile (A,B) based on the threshold using randomized algorithms for fixed precision.
    Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
    Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion,
    Numerical Linear Algebra with Applications 23 (2), pp. 314-339.
    --credit to: Umberto Villa
    """
    if dim is None:
        dim=Omega[0].size()
    
    nvec  = Omega.nvec()
    
    assert(nvec >= incr_k )
    
    Ybar = MultiVector(Omega[0], nvec)
    Y_pr = MultiVector(Omega)
    Q = MultiVector(Omega)
    for i in range(s):
        Y_pr.swap(Q)
        MatMvMult(A, Y_pr, Ybar)
        MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    BQ, _ = Q.Borthogonalize(B)
    
    Xt = Y_pr.dot_mv(BQ)
    Wt = Ybar.dot_mv(Q)
    Tt = np.linalg.solve(Xt,Wt)
                
    T = .5*Tt + .5*Tt.T
    
    d=np.zeros(0); #U_nd=np.zeros((0,dim))
    U=MultiVector(Omega[0], 0)
    num_eigs=0
    BX = MultiVector(Omega[0], nvec)
    while num_eigs<dim+np.float_(incr_k)/2:
        d_k,U_k=_lr_eig(Omega,T,incr_k,Q)
        # threshold
        idx = d_k>=thld
        d=np.append(d,d_k[idx])
        if any(idx):
#             U_nd=np.append(U_nd,[U_k[i] for i in np.where(idx)[0]],axis=0) # np.concatenate not appropriate for mpirun
            U_=MultiVector(U); nvec_=U_.nvec()
            U.setSizeFromVector(Omega[0],nvec_+np.sum(idx))
            for i in range(nvec_): U[i][:]=U_[i]
            for i,j in enumerate(np.where(idx)[0]): U[nvec_+i][:]=U_k[j]
        if sum(idx)<incr_k:
            break
        else:
            Q = MultiVector(Omega)
            for i in range(s):
                Y_pr.swap(Q)
                MatMvMult(B, Y_pr, BX)
                MvDSmatMult(U_k, d_k[:,None]*U_k.dot_mv(BX), BQ)
                MatMvMult(B, BQ, BX)
                Ybar.axpy(-np.ones(nvec),BX)
                MatMvMult(Solver2Operator(Binv), Ybar, Q)
            
            BQ, _ = Q.Borthogonalize(B)
            
            Xt = Y_pr.dot_mv(BQ)
            Wt = Ybar.dot_mv(Q)
            Tt = np.linalg.solve(Xt,Wt)
                        
            T = .5*Tt + .5*Tt.T
            
        num_eigs+=incr_k
#     l_d=len(d)
#     U=MultiVector(Omega[0], l_d)
#     for i in range(l_d):
#         U[i][:]=U_nd[i,:]
    
    if check:
        check_g(A,B, U, d)
    
    return d, U

def singlePassGx(A,B,invB,Omega,**kwargs):
    """
    Get partial generalized eigen-pairs of pencile (A,B) using singlePass randomized algorithm.
    """
    if 'k' in kwargs:
        eigs = singlePassG(A,B,invB,Omega,**kwargs)
    elif 'thld' in kwargs:
        eigs = singlePassG_prec(A,B,invB,Omega,**kwargs)
    else:
        print('warning: no matched algorithm found!')
        eigs=[None]*2
        pass
    return eigs

if __name__ == '__main__':
    seed=2017
    np.random.seed(seed)
    import sys
    sys.path.append( "../../" )
    from hippylib import *
    from RANS import RANS
    rans = RANS()
    rans.setup(seed)
    # read MAP parameter
    parameter = dl.Function(rans.pde.Vh[PARAMETER])
    map_file=dl.HDF5File(rans.mpi_comm, 'map_solution.h5', "r")
    map_file.read(parameter, "parameter")
    map_file.close()
    parameter=parameter.vector()
    # get HessApply
    _,_,_,_ = rans.get_geom(parameter,geom_ord=[0,1],src4init='map_solution')
    HessApply = rans._get_HessApply(parameter,GN_appx=True,MF_only=False)
    
    # generalized eigen-problem
    print('\nTesting generalized eigenvalue problem...')
#     # accurate eigen-solver
#     eigen = dl.SLEPcEigenSolver(HessApply,rans.prior.R)
#     eigen.parameters['problem_type']='gen_hermitian'
#     k0=50
#     eigen.solve(k0)
#     d_xct=np.zeros(k0); U_xct=MultiVector(parameter, k0)
#     for i in range(k0):
#         d_xct[i],_,U_xct[i][:],_=eigen.get_eigenpairs(i)
#     print('True (generalized) eigen-values:')
#     print(d_xct)
    # randomized eigen-solver
    k=20; p=10
    Omega = MultiVector(parameter, k+p)
    for i in xrange(k+p):
        Random.normal(Omega[i], 1., True)
    eigs=singlePassGx(HessApply,rans.prior.R,rans.prior.Rsolver,Omega,k=k)
    print('Fixed %d rank solution:' % k)
    print(eigs[0])
#     print(eigs[1])
    prec=1e-2
    eigs=singlePassGx(HessApply,rans.prior.R,rans.prior.Rsolver,Omega,thld=prec,incr_k=10)
    print('Fixed precision %e solution:' % prec)
    print(eigs[0])