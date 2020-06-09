
# A dictionary of options for dynamic stein variational methods
options = {
    # 1. options for the parameter type
    "type_parameter": "field",
    # field: the parameter is a spatial field
    # vector: the parameter is a vector of finite dimensions
    "is_projection": False,
    # if True, the parameter is the coefficient of a high-dimensional parameter projected into a subspace
    "type_projection": "hessian",
    # hessian: Hessian-based projection E[\nabla^2 \log \pi]
    # fisher: Fisher-based projection E[\nabla \log pi (\nabla \log \pi)^T]
    "is_precondition": False,
    # True: use Fisher or Hessian matrix to precondition SVGD
    "tol_projection": 1.e-1,
    # the tolerance for dimension truncation/projection, i.e., |\lambda_{j+1}| <= tol <= |\lambda_j|
    "reduce_tol_projection": 2,
    # reduce the projection tolerance by a factor of 10
    "coefficient_dimension": 10,
    # the number of the bases for projection, or the dimension of the projection coefficient vector
    "add_dimension": 2,
    # the # of dimensions to add when the training converges in the projected subspace

    # 2. options for initializing and dynamically adding particles in the optimization process
    "seed": 'random',
    # random: generate particles with random seed
    # fixed: generate particles with fixed seed
    "number_particles": 1,
    # number of initial particles used to construct the transport map
    "number_particles_add": 0,
    # number of additional particles not used in constructing the transport map, but used
    # e.g., to compute the normalization constant as independent samples pushed to posterior
    "add_number": 0,
    # if 0, do not add particles
    # the number of particles to be added sampled from Laplace distribution at each particle, 1, 2, 3, ...
    "add_step": 10,
    # add particles every n steps of optimization, n = add_step,
    # this should be changed to more suitable criteria
    "add_rule": 1,
    # if 1, add particles and use all of them to construct the transport map
    # if 2, add particles but only add the ones added in the previous step to construct the transport map
    # if 3, add particles but do NOT use them to construct the transport map

    # 3. options for the optimization method
    "type_optimization": "newtonSeparated",
    # newtonSeparated: Stein variational Newton with separated system
    # newtonCoupled: Stein variational Newton with coupled system
    # gradientDescent: Stein variational gradient decent
    "type_Hessian": "lumped",
    # "full": assemble the fully coupled Hessian system, only work for NewtonCoupled
    # "lumped": add the Hessian in the same row to the diagonal one, work for both NewtonCoupled and NewtonSeparated
    # "diagonal": only use the diagonal part of the Hessian system, work for both NewtonCoupled and NewtonSeparated

    # 4. options for the Hessian misfit term
    "type_approximation": "hessian",
    # hessian: use Hessian
    # fisher: use Fisher information matrix as approximation of the local and averaged Hessian
    "low_rank_Hessian": True,
    # True: solve the generalized eigenvalue problem (H, R), R is the inverse of prior covariance
    "rank_Hessian": 20,
    # the rank of the Hessian of the misfit term at each particle
    "rank_Hessian_tol": 1.e-1,
    # the tolerance to determine the # of ranks such that H \psi_i = \lambda_i R \psi_i, \lambda_i >= rank_Hessian_tol
    "low_rank_Hessian_average": True,
    # True: solve the generalized eigenvalue problem (H, R), R is the inverse of prior covariance
    "rank_Hessian_average": 20,
    # the rank of the Hessian of the misfit term averaged over all particles
    "rank_Hessian_average_tol": 1.e-1,
    # the tolerance to determine the # of ranks such that H \psi_i = \lambda_i R \psi_i, \lambda_i >= rank_Hessian_tol
    "gauss_newton_approx": False,
    # if True, use Gauss Newton approximation of the Hessian of the misfit term, to make the system well-posed to solve
    "max_iter_gauss_newton_approx": 10,
    # the maximum number n of optimization steps to use gauss_newton_approx
    # if not 0, "gauss_newton_approx" is automatically set to True before reaching n

    # 5. options for the kernel
    "type_metric": "posterior_separate",
    # "prior":  the prior covariance
    # "posterior_average": the average of the posterior covariance at all current particles;
    # "posterior_separate": the posterior covariance at each of the current particle;
    "type_scaling": 1,
    # 0: no scaling
    # 1: scale the metric by the parameter dimension
    # 2: scale the metric by the trace of the negative log posterior
    # 3: scale the metric by the mean of the particle distances
    # 4: scale the metric by the mean of the balanced posterior gradient and kernel gradient
    "kernel_vectorized": False,
    # if True, compute kernel using vectorized structure
    "save_kernel": True,
    # if True, save the evaluations of the kernel and its gradient at every particle
    # set False for large numbers of both parameter dimension and particles, due to prohibitive memory usage
    "delta_kernel": False,
    # if True, set k(pn, pm) = 0 for pn != pm, and k(pn, pn) = 1, in order to accelerate moving particles to posterior
    "max_iter_delta_kernel": 0,
    # the maximum number n of optimization steps to use delta kernel
    # if not 0, "delta_kernel" is automatically set to True before reaching n

    # 6. options for saving data and stopping optimization
    "plot": False,
    # plot figures during construction
    "save_step": 10,
    # save data every n steps
    "save_number": 0,
    # the number of particles, kernels, eigenvalues to be generated and saved, if 0, do not save
    "rel_tolerance": 1e-6,
    # stop when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance
    "abs_tolerance": 1e-12,
    # stop when sqrt(g,g) <= abs_tolerance
    "gdm_tolerance": 1e-18,
    # stop when (g,dm) <= gdm_tolerance
    "step_tolerance": 1e-3,
    # stop when (dm, dm) * step_size <= 1e-3
    "step_projection_tolerance": 1e-2,
    # update projected bases when (dm, dm) * step_size <= 1e-3, which should be larger than step_tolerance
    "reduce_step_projection_tolerance": 2,
    # reduce the step_projection_tolerance by a factor of 2
    "inner_rel_tolerance": 1e-9,
    # relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems
    "max_iter": 100,
    # maximum number of iterations for the optimization
    "cg_coarse_tolerance": .5e-2,
    # Coarsest tolerance for the CG method (Eisenstat-Walker)
    "line_search": True,
    # do line search if True
    "search_size": 1.,
    # step size to start the line search
    "c_armijo": 1e-4,
    # Armijo constant for sufficient reduction
    "max_backtracking_iter": 10,
    # Maximum number of backtracking iterations
    "print_level": -1,
    # Control verbosity of printing screen
    "termination_reasons": [
        "Maximum number of Iteration reached",  # 0
        "Norm of the gradient less than tolerance",  # 1
        "Maximum number of backtracking reached",  # 2
        "Norm of (g, dm) less than tolerance",  # 3
        "Norm of alpha * dm less than tolerance"  # 4
    ]
    # the reasons for terminating the optimization
}