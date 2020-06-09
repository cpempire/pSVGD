from __future__ import absolute_import, division, print_function

import numpy as np
from mpi4py import MPI


def orthogonalize(Q):
    n = Q.shape[1]
    R = np.zeros((n, n), dtype='d')
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
                R[i, k] += s
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
        R[k, k] = tt
        if np.abs(tt * eps) > 0.:
            tt = 1. / tt
        else:
            tt = 0.

        Q[:,k] *= tt

    return Q, R


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
    R = np.zeros((n, n), dtype='d')
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
                R[i, k] += s
                Q[:,k] -= s*Q[:,i]

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
        R[k, k] = tt
        if np.abs(tt * eps) > 0.:
            tt = 1. / tt
        else:
            tt = 0.

        Q[:,k] *= tt
        Bq[:,k] *= tt

    return Q, R


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

    nvec = Omega.shape[1]

    Ybar = np.zeros_like(Omega)
    Q = np.copy(Omega)
    for i in range(s):
        for j in range(nvec):
            A.mult(Q[:, j], Ybar[:, j])
        Q = np.copy(Ybar)

    Q, _ = orthogonalize(Q)

    AQ = np.zeros_like(Q)
    for j in range(nvec):
        A.mult(Q[:, j], AQ[:, j])

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


def doublePassG(A, B, Binv, Omega, k, s=1, check=False):

    nvec = Omega.shape[1]

    Ybar = np.zeros_like(Omega)
    Q = np.copy(Omega)
    for i in range(s):
        for j in range(nvec):
            A.mult(Q[:, j], Ybar[:, j])
            Binv.mult(Ybar[:, j], Q[:, j])

    Q, _ = Borthogonalize(Q, B)

    AQ = np.zeros_like(Q)
    for j in range(Q.shape[1]):
        A.mult(Q[:, j], AQ[:, j])

    T = np.dot(Q.T, AQ)

    d, V = np.linalg.eigh(T)
    sort_perm = np.abs(d).argsort()

    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]]

    U = np.dot(Q, V)

    # print("d = ", d,
    #       "\nV = ", V,
    #       "\nU = ", U)

    if check:
        check_g(A, B, U, d)

    return d, U


def check_std(A, U, d):
    """
    Test the frobenious norm of  :math:`U^TU - I_k`.

    Test the frobenious norm of  :math:`(V^TAV) - I_k`, with :math:`V = U D^{-1/2}`.

    Test the :math:`l_2` norm of the residual: :math:`r[i] = A U[i] - d[i] U[i]`.
    """
    nvec = U.shape[1]
    AU = np.zeros_like(U)
    for i in range(nvec):
        A.mult(U[:, i], AU[:, i])

    # Residual checks
    diff = np.copy(AU)
    diff -= np.dot(U, np.diag(d))

    # ortho check
    UtU = np.dot(U.T, U)
    err = UtU - np.eye(nvec, dtype=UtU.dtype)
    err_Bortho = np.linalg.norm(err, 'fro')

    # A-ortho check
    V = np.copy(U)
    scaling = np.power(np.abs(d), -0.5)
    V = np.dot(V, np.diag(scaling))
    AV = np.dot(AU, np.diag(scaling))
    VtAV = np.dot(V.T, AV)
    err = VtAV - np.diag(np.sign(d))
    err_Aortho = np.linalg.norm(err, 'fro')

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print( "|| UtU - I ||_F = ", err_Bortho)
        print( "|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print( "lambda", "||Au - lambda*u||_2")
        for i in range(nvec):
            print("{0:5e} {1:5e}".format(d[i], np.linalg.norm(diff[:,i])))


def check_g(A, B, U, d):
    """
    Test the frobenious norm of  :math:`U^TBU - I_k`.

    Test the frobenious norm of  :math:`(V^TAV) - I_k`, with :math:`V = U D^{-1/2}`.

    Test the :math:`l_2` norm of the residual: :math:`r[i] = A U[i] - d[i] B U[i]`.
    """
    nvec = U.shape[1]
    AU = np.zeros_like(U)
    BU = np.zeros_like(U)
    for i in range(nvec):
        A.mult(U[:, i], AU[:, i])
        B.mult(U[:, i], BU[:, i])

    # Residual checks
    diff = np.copy(AU)
    diff -= np.dot(BU, np.diag(d))

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
        print("|| UtBU - I ||_F = ", err_Bortho)
        print("|| VtAV - I ||_F = ", err_Aortho, " with V = U D^{-1/2}")
        print("lambda", "||Au - lambdaBu||_2")
        for i in range(nvec):
            print("{0:5e} {1:5e}".format(d[i], np.linalg.norm(diff[:,i])))
