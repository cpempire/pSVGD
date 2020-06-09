from model import *
#
# from sine import *

import os
import time
import pickle

# check the stein/options to see all possible choices
options["type_parameter"] = 'vector'
options["type_optimization"] = "gradientDecent"
options["is_projection"] = True
options["update_step"] = 10
options["type_projection"] = "fisher"
options["tol_projection"] = 1.e-1
options["coefficient_dimension"] = 10
options["add_dimension"] = 2

options["number_particles"] = 8
options["number_particles_add"] = 0
options["add_number"] = 0
options["add_step"] = 5
options["add_rule"] = 1

options["type_Hessian"] = "lumped"

options["type_scaling"] = 1
options["type_metric"] = "posterior_average"
# options["type_metric"] = "prior"
options["kernel_vectorized"] = True

options["low_rank_Hessian"] = True
options["rank_Hessian"] = np.min([options["number_particles"]*comm.Get_size(), 50])
options["rank_Hessian_tol"] = 1.e-2
options["low_rank_Hessian_average"] = True
options["rank_Hessian_average"] = np.min([options["number_particles"]*comm.Get_size(), 50])
options["rank_Hessian_average_tol"] = 1.e-2
options["gauss_newton_approx"] = False  # if error of unable to solve linear system occurs, use True

options["max_iter"] = 201
options["save_number"] = options["number_particles"]
options["step_tolerance"] = 1.e-7
options["step_projection_tolerance"] = 1.e-2
options["line_search"] = True
options["search_size"] = 1.e-0
options["max_backtracking_iter"] = 10
options["cg_coarse_tolerance"] = 0.5e-2
options["print_level"] = 3
options["plot"] = False
options["save_step"] = 10

# dimension = 101
# prior = BrownianMotion(dimension)
#
# if os.path.isfile("data/observation.p"):
#     data = pickle.load(open("data/observation.p", "rb"))
#     dimension, beta, x, u, obs, loc, noise_covariance = data
# else:
#     sample_noise = np.random.normal(0, 1, dimension)
#     x = np.zeros(dimension)
#     prior.sample(sample_noise, x)
#
#     beta = 10.
#     misfit = Misfit(dimension, beta)
#     u = misfit.solution(x)
#     loc = np.arange(0, dimension, 5)
#     sigma = 0.1
#     obs_noise = np.random.normal(0, sigma, len(loc))
#     obs = u[loc] + obs_noise
#     noise_covariance = np.diag(sigma**2*np.ones(len(loc)))
#
#     data = (dimension, beta, x, u, obs, loc, noise_covariance)
#     pickle.dump(data, open("data/observation.p", 'wb'))
#
# misfit = Misfit(dimension, beta, obs, loc, noise_covariance)
#
# model = Model(prior, misfit)

# generate particles
particle = Particle(model, options, comm)

# filename = "data/laplace.p"
filename = "data/mcmc_dili_sample.p"
if os.path.isfile(filename):
    print("set reference for mean and variance")
    data_save = pickle.load(open(filename, 'rb'))
    mean = model.generate_vector()
    mean.set_local(data_save["mean"])
    variance = model.generate_vector()
    variance.set_local(data_save["variance"])
    particle.mean_posterior = mean
    particle.variance_posterior = variance

# evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
variation = Variation(model, particle, options, comm)
variation.update(particle)
if options["is_projection"]:
    particle.projection()

# evaluate the kernel and its gradient at given particles
kernel = Kernel(model, particle, variation, options, comm)

t0 = time.time()

solver = GradientDescent(model, particle, variation, kernel, options, comm)

solver.solve()

print("GradientDecent solving time = ", time.time() - t0)

np.savez("data/social-distancing-pSVGD", control=model.misfit.controls)
