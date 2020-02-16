from model_linear import *

import time
import pickle

# check the stein/options to see all possible choices
options["is_projection"] = False
options["type_projection"] = "hessian"
options["coefficient_dimension"] = 50
options["add_dimension"] = 5

options["number_particles"] = 1
options["number_particles_add"] = 0
options["add_number"] = 0
options["add_step"] = 5
options["add_rule"] = 1

options["type_Hessian"] = "lumped"

options["type_scaling"] = 1
options["type_metric"] = "posterior_average"

options["low_rank_Hessian"] = True
options["rank_Hessian"] = 50
options["rank_Hessian_tol"] = 1.e-3
options["low_rank_Hessian_average"] = True
options["rank_Hessian_average"] = 50
options["rank_Hessian_average_tol"] = 1.e-3
options["gauss_newton_approx"] = True  # if error of unable to solve linear system occurs, use True

options["step_tolerance"] = 1.e-16
options["step_projection_tolerance"] = 1.e-8
options["max_iter"] = 100
options["line_search"] = True
options["max_backtracking_iter"] = 10
options["cg_coarse_tolerance"] = 0.5e-2
options["print_level"] = -1
options["plot"] = True

# generate particles
particle = Particle(model, options, comm)

# evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
variation = Variation(model, particle, options, comm)

# evaluate the kernel and its gradient at given particles
kernel = Kernel(model, particle, variation, options, comm)

t0 = time.time()

solver = NewtonSeparated(model, particle, variation, kernel, options, comm)

solver.solve()

print("NewtonSeparated solving time = ", time.time() - t0)

# compute the mean and variance of Laplace approximation at MAP point
if nproc == 1 and particle.number_particles == 1:

    data_save = dict()

    mean = particle.particles[0]
    mhelp = model.generate_vector(PARAMETER)
    model.prior.M.mult(mean, mhelp)
    mean_norm = np.sqrt(mean.inner(mhelp))

    posterior = GaussianLRPosterior(model.prior, variation.d[0], variation.U[0], mean=particle.particles[0])

    rank_prior = 200
    post_pointwise_variance, _, _ = posterior.pointwise_variance(method="Randomized", r=rank_prior)
    phelp = model.generate_vector(PARAMETER)
    model.prior.M.mult(post_pointwise_variance, phelp)
    variance_norm = np.sqrt(post_pointwise_variance.inner(phelp))

    trace = np.sum(post_pointwise_variance.get_local())

    print("for Laplace approximation at MAP point ",
          " mean L2-norm = ", mean_norm, " pointwise variance L2-norm = ", variance_norm, "trace of covariance = ", trace)

    true_statistics = [mean_norm, variance_norm, trace]
    data_save["true_statistics"] = true_statistics
    data_save["mean"] = mean.get_local()
    data_save["variance"] = post_pointwise_variance.get_local()
    data_save["sample_statistics"] = []
    data_save["qoi_statistics"] = []
    data_save["cost_statistics"] = []

    N = 512
    print("samples from Laplace approximation")
    sampler = GaussianLRPosterior(model.prior, variation.d[0], variation.U[0], mean=particle.particles[0])
    noise = dl.Vector()
    model.prior.init_vector(noise, "noise")
    samples = []
    qoi = []
    cost = []
    for m in range(N):
        s_prior = model.generate_vector(PARAMETER)
        s_posterior = model.generate_vector(PARAMETER)
        parRandom.normal(1., noise)
        model.prior.sample(noise, s_prior, add_mean=False)
        sampler.sample(s_prior, s_posterior, add_mean=True)
        samples.append(s_posterior)
        x_star = model.generate_vector()
        # update the parameter
        x_star[PARAMETER].zero()
        x_star[PARAMETER].axpy(1., s_posterior)
        # update the state at new parameter
        x_star[STATE].zero()
        model.solveFwd(x_star[STATE], x_star)

        # evaluate the cost functional, here the potential
        cos, reg, misfit = model.cost(x_star)
        print("posterior cost, reg, misfit = ", cos, reg, misfit)
        cost.append(cos)

        # evaluate qoi
        qoi.append(model.qoi.eval(x_star))

        s_prior = model.generate_vector(PARAMETER)
        model.prior.sample(noise, s_prior, add_mean=True)
        x_star[PARAMETER].zero()
        x_star[PARAMETER].axpy(1., s_prior)
        # update the state at new parameter
        x_star[STATE].zero()
        model.solveFwd(x_star[STATE], x_star)

        # evaluate the cost functional, here the potential
        cos, reg, misfit = model.cost(x_star)
        print("prior cost, reg, misfit = ", cos, reg, misfit)

    sample_statistics = []
    qoi_statistics = []
    cost_statistics = []
    for n in [2, 8, 32, 128, 512]:
        mean = model.generate_vector(PARAMETER)
        mhelp = model.generate_vector(PARAMETER)
        for m in range(n):
            mean.axpy(1./n, samples[m])
        model.prior.M.mult(mean, mhelp)
        mean_norm = np.sqrt(mean.inner(mhelp))

        variance = model.generate_vector(PARAMETER)
        vhelp = model.generate_vector(PARAMETER)
        for m in range(n):
            vhelp = model.generate_vector(PARAMETER)
            vhelp.axpy(1.0, samples[m])
            vhelp.axpy(-1.0, mean)
            vsquared = vhelp.get_local() ** 2
            vhelp.set_local(vsquared)
            variance.axpy(1.0 / (n-1), vhelp)

        trace = np.sum(variance.get_local())
        vhelp = model.generate_vector(PARAMETER)
        model.prior.M.mult(variance, vhelp)
        variance_norm = np.sqrt(variance.inner(vhelp))
        qoi_mean = np.mean(qoi[:n])
        qoi_std = np.std(qoi[:n])
        cost_mean = np.mean(cost[:n])
        cost_std = np.std(cost[:n])

        print("for Laplace sampling at MAP with # samples = ", n,
              " mean L2-norm = ", mean_norm, " pointwise variance L2-norm = ", variance_norm,
              "trace of covariance = ", trace, "qoi_mean = ", qoi_mean, "qoi_std", qoi_std,
              "cost_mean = ", cost_mean, "cost_std = ", cost_std)

        sample_statistics.append([n, mean_norm, variance_norm, trace])
        qoi_statistics.append([n, qoi_mean, qoi_std])
        cost_statistics.append([n, cost_mean, cost_std])
    data_save["sample_statistics"] = sample_statistics
    data_save["qoi_statistics"] = qoi_statistics
    data_save["cost_statistics"] = cost_statistics

    filename = "data/laplace_nDimension_"+str(particle.particle_dimension)+".p"
    pickle.dump(data_save, open(filename, "wb"))
