from model_lognormal_32 import *

import time
import pickle

# check the stein/options to see all possible choices
options["type_optimization"] = "gradientDescent"
options["is_projection"] = True
options["tol_projection"] = 1.e-2
options["type_projection"] = "fisher"
options["is_precondition"] = False
options["type_approximation"] = "fisher"
options["coefficient_dimension"] = 50
options["add_dimension"] = 0

options["number_particles"] = 512
options["number_particles_add"] = 0
options["add_number"] = 0
options["add_step"] = 5
options["add_rule"] = 1

options["type_scaling"] = 1
options["type_metric"] = "posterior_average"  # posterior_average

options["type_Hessian"] = "lumped"
options["low_rank_Hessian"] = False
options["rank_Hessian"] = 50
options["rank_Hessian_tol"] = 1.e-2
options["low_rank_Hessian_average"] = False
options["rank_Hessian_average"] = 50
options["rank_Hessian_average_tol"] = 1.e-2
options["gauss_newton_approx"] = True  # if error of unable to solve linear system occurs, use True

options["max_iter"] = 2
options["step_tolerance"] = 1e-7
options["step_projection_tolerance"] = 1e-3
options["line_search"] = False
options["search_size"] = 1e-5
options["max_backtracking_iter"] = 10
options["cg_coarse_tolerance"] = 0.5e-2
options["print_level"] = -1
options["save_number"] = 0
options["plot"] = False

# generate particles
particle = Particle(model, options, comm)

# evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
variation = Variation(model, particle, options, comm)

# evaluate the kernel and its gradient at given particles
kernel = Kernel(model, particle, variation, options, comm)

t0 = time.time()

solver = GradientDescent(model, particle, variation, kernel, options, comm)

solver.solve()

time_total = time.time() - t0

print("GradientDecent solving time = ", time_total)


time_communication = [particle.time_communication, variation.time_communication, kernel.time_communication, solver.time_communication]
time_computation = [particle.time_computation, variation.time_computation, kernel.time_computation, solver.time_computation]
time_update_bases = [variation.time_update_bases_communication, variation.time_update_bases_computation, variation.number_update_bases]

print("time_communication for [particle, variation, kernel, solver] = ", time_communication,
      "time_computation for [particle, variation, kernel, solver] = ", time_computation,
      "time_update_bases for [communication, computation, times] = ", time_update_bases)

data_save = dict()
data_save["time_communication"] = time_communication
data_save["time_computation"] = time_computation
data_save["time_update_bases"] = time_update_bases
data_save["time_total"] = time_total

filename = "data/time_nCore_"+str(nproc)+".p"
pickle.dump(data_save, open(filename, 'wb'))
