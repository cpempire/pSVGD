from model_lognormal import *
import matplotlib.pyplot as plt
import numpy as np

sep = "\n"+"#"*80+"\n"
if rank == 0:
    print(sep, "Find the MAP point", sep)
m = prior.mean.copy()
parameters = ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1e-8
parameters["abs_tolerance"] = 1e-12
parameters["max_iter"] = 25
parameters["inner_rel_tolerance"] = 1e-15
parameters["globalization"] = "LS"
parameters["GN_iter"] = 5
if rank != 0:
    parameters["print_level"] = -1
solver = ReducedSpaceNewtonCG(model, parameters)

x = solver.solve([None, m, None])

if rank == 0:
    if solver.converged:
        print("\nConverged in ", solver.it, " iterations.")
    else:
        print("\nNot Converged")

    print("Termination reason: ", solver.termination_reasons[solver.reason])
    print("Final gradient norm: ", solver.final_grad_norm)
    print("Final cost: ", solver.final_cost)

## Build the low rank approximation of the posterior
model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
Hmisfit = ReducedHessian(model, solver.parameters["inner_rel_tolerance"], misfit_only=True)
k = 50
p = 20
if rank == 0:
    print("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k, p))

Omega = MultiVector(x[PARAMETER], k + p)
parRandom.normal(1., Omega)

d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=2, check=False)
d[d < 0.] = 0.
nu = GaussianLRPosterior(prior, d, U)
nu.mean = x[PARAMETER]

noise = dl.Vector()
nu.init_vector(noise, "noise")
parRandom.normal(1., noise)
pr_s = model.generate_vector(PARAMETER)
post_s = model.generate_vector(PARAMETER)

nu.sample(noise, pr_s, post_s, add_mean=True)

# kMALA = MALAKernel(model)
# kMALA.parameters["delta_t"] = 2.5e-4

# kpCN = pCNKernel(model)
# kpCN.parameters["s"] = 0.025

# kgpCN = gpCNKernel(model, nu)
# kgpCN.parameters["s"] = 0.5

# klpCN = lpCNKernel(model)
# klpCN.parameters["s"] = 0.5

# kIS = ISKernel(model, nu)
#
kSN = SNKernel(model)

# kSNmap = SNmapKernel(model, nu)

# for kernel in [kpCN]:
# for kernel in [kgpCN]:
# for kernel in [klpCN]:
# for kernel in [kMALA]:
for kernel in [kSN]:
# for kernel in [kSNmap]:

    if rank == 0:
        print(kernel.name())

    # fid_m = dl.File(kernel.name()+"/parameter.pvd")
    # fid_u = dl.File(kernel.name()+"/state.pvd")
    fid_m = None
    fid_u = None
    chain = MCMC(kernel)
    chain.parameters["burn_in"] = 0
    chain.parameters["number_of_samples"] = 100
    chain.parameters["print_progress"] = 10
    tracer = FullTracer(chain.parameters["number_of_samples"], Vh, fid_m, fid_u, model)
    if rank != 0:
        chain.parameters["print_level"] = -1

    n_accept = chain.run(post_s, qoi, tracer)

    iact, lags, acoors = integratedAutocorrelationTime(tracer.data[:, 1])

    sample_1st_moment = model.generate_vector(PARAMETER)
    sample_2nd_moment = model.generate_vector(PARAMETER)

    filename = 'data/sample_1st_moment.xdmf'
    particle_fun = dl.Function(Vh[PARAMETER], name='particle')
    particle_fun.vector().axpy(1.0, tracer.m)
    if dlversion() <= (1, 6, 0):
        dl.File(mesh.mpi_comm(), filename) << particle_fun
    else:
        xf = dl.XDMFFile(mesh.mpi_comm(), filename)
        xf.write(particle_fun)

    filename = 'data/sample_2nd_moment.xdmf'
    particle_fun = dl.Function(Vh[PARAMETER], name='particle')
    particle_fun.vector().axpy(1.0, tracer.m2)
    if dlversion() <= (1, 6, 0):
        dl.File(mesh.mpi_comm(), filename) << particle_fun
    else:
        xf = dl.XDMFFile(mesh.mpi_comm(), filename)
        xf.write(particle_fun)

    if rank == 0:
        pickle.dump(tracer.data, open("data/mcmc.p", 'wb'))
        # np.savetxt(kernel.name() + ".txt", tracer.data)
        print("Number accepted = {0}".format(n_accept))
        print("E[qoi] = {0}".format(np.mean(tracer.data[:,0])))
        print("Std[qoi] = {0}".format(np.std(tracer.data[:,0])))
        print("E[cost] = {0}".format(np.mean(tracer.data[:,1])))
        print("Std[cost] = {0}".format(np.std(tracer.data[:,1])))

        fig = plt.figure()
        plt.plot(tracer.data[:, 0], 'b.')

        print("Integrated autocorrelation time = {0}".format(iact))

if rank == 0:
    plt.show()
    filename = "figure/mcmc.pdf"
    fig.savefig(filename, format='pdf')
    filename = "figure/mcmc.eps"
    fig.savefig(filename, format='eps')
