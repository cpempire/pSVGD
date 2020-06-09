
from __future__ import absolute_import, division, print_function

import numpy as np
from mpi4py import MPI

"""
Randomized algorithms for the solution of Hermitian Eigenvalues Problems (HEP)
and Generalized Hermitian Eigenvalues Problems (GHEP).

In particular we provide an implementation of the single and double pass algorithms
and some convergence test.

REFERENCES:

Nathan Halko, Per Gunnar Martinsson, and Joel A. Tropp,
Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions,
SIAM Review, 53 (2011), pp. 217-288.

Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application
to computing Karhunen-Loeve expansion,
Numerical Linear Algebra with Applications, to appear.
"""


def orthogonalize(Q):
    n = Q.shape[1]
    r = np.zeros((n, n), dtype='d')
    reorth = np.zeros((n,), dtype='d')
    eps = np.finfo(np.float64).eps

    for k in np.arange(n):
        t = np.sqrt(Q[:,k].dot(Q[:,k]))

        nach = 1
        u = 0
        while nach:
            u += 1
            for i in np.arange(k):
                s = Q[:,i].dot(Q[:,k])
                r[i, k] += s
                Q[:,k] -= s * Q[:,i]

            tt = np.sqrt(Q[:,k].dot(Q[:,k]))
            if tt > t * 10. * eps and tt < t / 10.:
                nach = 1
                t = tt
            else:
                nach = 0
                if tt < 10. * eps * t:
                    tt = 0.

        reorth[k] = u
        r[k, k] = tt
        if np.abs(tt * eps) > 0.:
            tt = 1. / tt
        else:
            tt = 0.

        Q[:,k] *= tt

    return Q


def Borthogonalize(Q, B):
    """
    Returns :math:`QR` decomposition of self, which satisfies conditions (1)--(4).
    Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
    for computing the :math:`B`-orthogonal :math:`QR` factorization.

    References:
        1. `A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized \
        Hermitian Eigenvalue Problems with application to computing \
        Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885`
        2. `W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980`

    https://github.com/arvindks/kle

    """
    n = Q.shape[1]
    Bq = np.zeros_like(Q)
    r = np.zeros((n, n), dtype='d')
    reorth = np.zeros((n,), dtype='d')
    eps = np.finfo(np.float64).eps

    for k in np.arange(n):
        B.mult(Q[:,k], Bq[:,k])
        t = np.sqrt(Bq[:,k].dot(Q[:,k]))

        nach = 1
        u = 0
        while nach:
            u += 1
            for i in np.arange(k):
                s = Bq[:,i].dot(Q[:,k])
                r[i, k] += s
                Q[:,k] += -s*Q[:,i]

            B.mult(Q[:,k], Bq[:,k])
            tt = np.sqrt(Bq[:,k].dot(Q[:,k]))
            if tt > t * 10. * eps and tt < t / 10.:
                nach = 1
                t = tt
            else:
                nach = 0
                if tt < 10. * eps * t:
                    tt = 0.

        reorth[k] = u
        r[k, k] = tt
        if np.abs(tt * eps) > 0.:
            tt = 1. / tt
        else:
            tt = 0.

        Q[:,k] *= tt
        Bq[:,k] *= tt

    # return Bq, r
    return Q

# def singlePass(A,Omega,k,s=1,check=False):
#     """
#     The single pass algorithm for the Hermitian Eigenvalues Problems (HEP) as presented in [1].
#
#     Inputs:
#
#     - :code:`A`: the operator for which we need to estimate the dominant eigenpairs.
#     - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
#     - :code:`k`: the number of eigenpairs to extract.
#
#     Outputs:
#
#     - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
#     - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T U = I_k`.
#     """
#     nvec  = Omega.nvec()
#     assert(nvec >= k )
#
#     Y_pr = np.copy(Omega)
#     Y = np.copy(Omega)
#     for i in range(s):
#         Y_pr.swap(Y)
#         MatMvMult(A, Y_pr, Y)
#
#     Q = np.copy(Y)
#     Q.orthogonalize()
#
#     Zt = Y_pr.dot_mv(Q)
#     Wt = Y.dot_mv(Q)
#
#     Tt = np.linalg.solve(Zt, Wt)
#
#     T = .5*Tt + .5*Tt.T
#
#     d, V = np.linalg.eigh(T)
#     sort_perm = np.abs(d).argsort()
#
#     sort_perm = sort_perm[::-1]
#     d = d[sort_perm[0:k]]
#     V = V[:, sort_perm[0:k]]
#
#     U = np.copy(Omega[0], k)
#     MvDSmatMult(Q, V, U)
#
#     if check:
#         check_std(A, U, d)
#
#     return d, U


def doublePass(A, Omega, k, s=1, check=False):
    """
    The double pass algorithm for the HEP as presented in [1].

    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant eigenpairs.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.

    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T U = I_k`.
    """

    nvec = len(Omega)
    assert (nvec >= k)

    Y = np.zeros_like(Omega)
    Q = np.copy(Omega)
    for i in range(s):
        for j in range(nvec):
            A.mult(Q[j], Y[j])  # todo make Hessian action in a list of vectors
    # Q.swap(Y)
    Q = orthogonalize(Q)

    AQ = np.zeros_like(Omega)
    for j in range(nvec):
        A.mult(Q[j], AQ[j])

    T = np.dot(Q.T, AQ)

    d, V = np.linalg.eigh(T)
    sort_perm = np.abs(d).argsort()

    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]]

    U = np.dot(Q, V)

    if check:
        check_std(A, U, d)

    return d, U

# def singlePassG(A, B, Binv, Omega,k, s = 1, check = False):
#     """
#     The single pass algorithm for the Generalized Hermitian Eigenvalues Problems (GHEP) as presented in [2].
#
#     Inputs:
#
#     - :code:`A`: the operator for which we need to estimate the dominant generalized eigenpairs.
#     - :code:`B`: the right-hand side operator.
#     - :code:`Binv`: the inverse of the right-hand side operator.
#     - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
#     - :code:`k`: the number of eigenpairs to extract.
#     - :code:`s`: the number of power iterations for selecting the subspace.
#
#     Outputs:
#
#     - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
#     - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T B U = I_k`.
#     """
#     nvec  = Omega.nvec()
#
#     assert(nvec >= k )
#
#     Ybar = np.copy(Omega[0], nvec)
#     Y_pr = np.copy(Omega)
#     Q = np.copy(Omega)
#     for i in range(s):
#         Y_pr.swap(Q)
#         MatMvMult(A, Y_pr, Ybar)
#         MatMvMult(Solver2Operator(Binv), Ybar, Q)
#
#     BQ, _ = Q.Borthogonalize(B)
#
#     Xt = Y_pr.dot_mv(BQ)
#     Wt = Ybar.dot_mv(Q)
#     Tt = np.linalg.solve(Xt,Wt)
#
#     T = .5*Tt + .5*Tt.T
#
#     d, V = np.linalg.eigh(T)
#     sort_perm = np.abs(d).argsort()
#
#     sort_perm = sort_perm[::-1]
#     d = d[sort_perm[0:k]]
#     V = V[:, sort_perm[0:k]]
#
#     U = np.copy(Omega[0], k)
#     MvDSmatMult(Q, V, U)
#
#     if check:
#         check_g(A,B, U, d)
#
#     return d, U


def doublePassG(A, B, Binv, Omega, k, s=1, check=True):
    """
    The double pass algorithm for the GHEP as presented in [2].
    
    Inputs:

    - :code:`A`: the operator for which we need to estimate the dominant generalized eigenpairs.
    - :code:`B`: the right-hand side operator.
    - :code:`Binv`: the inverse of the right-hand side operator.
    - :code:`Omega`: a random gassian matrix with :math:`m \\geq k` columns.
    - :code:`k`: the number of eigenpairs to extract.
    - :code:`s`: the number of power iterations for selecting the subspace.
    
    Outputs:

    - :code:`d`: the estimate of the :math:`k` dominant eigenvalues of :math:`A`.
    - :code:`U`: the estimate of the :math:`k` dominant eigenvectors of :math:`A,\\, U^T B U = I_k`.
    """        
    nvec = len(Omega)
    assert(nvec >= k)
    
    Ybar = np.zeros_like(Omega)
    Q = np.copy(Omega)
    for i in range(s):
        for j in range(nvec):
            A.mult(Q[j], Ybar[j])     # todo make Hessian action in a list of vectors
            Binv.mult(Ybar[j], Q[j])
    
    # Q = Borthogonalize(Q, B)
    # AQ = np.zeros_like(Omega)
    Q = Borthogonalize(Q.T, B)
    print("Q = ", Q)
    AQ = np.zeros_like(Q)
    for j in range(Q.shape[1]):
        A.mult(Q[:,j], AQ[:,j])

    T = np.dot(Q.T, AQ)
                        
    d, V = np.linalg.eigh(T)
    sort_perm = np.abs(d).argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 

    # print("d, V = ", d, V)
    # print("Q, V = ", Q, V)

    # kmin = np.min([k, Omega.shape[1]])
    # U = np.dot(Q[:kmin], V)
    # U = U[:k, :]
    U = np.dot(Q, V)
    print("d = ", d,
          "\nV = ", V,
          "\nU = ", U)

    # print("k, U = ", k, U, len(U))
    # MvDSmatMult(Q, V, U)
    
    if check:
        check_g(A, B, U, d)
            
    return d, U


def check_std(A, U, d):
    """
    Test the frobenious norm of  :math:`U^TU - I_k`.

    Test the frobenious norm of  :math:`(V^TAV) - I_k`, with :math:`V = U D^{-1/2}`.

    Test the :math:`l_2` norm of the residual: :math:`r[i] = A U[i] - d[i] U[i]`.
    """
    nvec  = len(U)
    AU = np.zeros_like(U)
    A.mult(U, AU)

    # Residual checks
    diff = np.copy(AU)
    diff -= np.dot(U, np.diag(d))
    res_norms = np.linalg.norm(diff, 2)

    # B-ortho check
    UtU = np.dot(U.T, U)
    err = UtU - np.eye(nvec, dtype=UtU.dtype)
    err_Bortho = np.linalg.norm(err, 'fro')

    #A-ortho check
    V = np.copy(U)
    scaling = np.power(np.abs(d), -0.5)
    V = np.dot(V, np.diag(scaling))
    AV = np.dot(AU, np.diag(scaling))
    VtAV = np.dot(V.T, AV)
    err = VtAV - np.diag(np.sign(d))#np.eye(nvec, dtype=VtAV.dtype)
    err_Aortho = np.linalg.norm(err, 'fro')

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print( "|| UtU - I ||_F = ", err_Bortho)
        print( "|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print( "lambda", "||Au - lambda*u||_2")
        for i in range(res_norms.shape[0]):
            print( "{0:5e} {1:5e}".format(d[i], res_norms[i]))


def check_g(A, B, U, d):
    """
    Test the frobenious norm of  :math:`U^TBU - I_k`.

    Test the frobenious norm of  :math:`(V^TAV) - I_k`, with :math:`V = U D^{-1/2}`.

    Test the :math:`l_2` norm of the residual: :math:`r[i] = A U[i] - d[i] B U[i]`.
    """
    nvec  = U.shape[1]
    AU = np.zeros_like(U)
    BU = np.zeros_like(U)
    for i in range(nvec):
        A.mult(U[:,i], AU[:,i])
        B.mult(U[:,i], BU[:,i])

    # Residual checks
    diff = np.copy(AU)
    diff -= np.dot(BU, np.diag(d))
    res_norms = np.linalg.norm(diff, 2)
    
    # B-ortho check
    UtBU = np.dot(U.T, BU)
    err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
    err_Bortho = np.linalg.norm(err, 'fro')
    
    # A-ortho check
    V = np.copy(U)
    scaling = np.power(np.abs(d), -0.5)
    V = np.dot(V, np.diag(scaling))
    AV = np.dot(AU, np.diag(scaling))
    VtAV = np.dot(V.T, AV)
    err = VtAV - np.eye(nvec, dtype=VtAV.dtype)
    err_Aortho = np.linalg.norm(err, 'fro')

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print( "|| UtBU - I ||_F = ", err_Bortho)
        print( "|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print( "lambda", "||Au - lambdaBu||_2")
        for i in range(res_norms.shape[0]):
            print( "{0:5e} {1:5e}".format(d[i], res_norms[i]) )
