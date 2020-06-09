
from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import dolfin as dl
from ..modeling.variables import PARAMETER
from ..utils.random import Random
from ..modeling.posterior import GaussianLRPosterior
from ..utils.checkDolfinVersion import dlversion

plot_valid = True
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except:
    plot_valid = False
    print("can not import pyplot")

if plot_valid:
    if not os.path.isdir("figure"):
        os.mkdir("figure")

if not os.path.isdir("data"):
    os.mkdir("data")


class Particle:
    # class to generate, move, and add particles
    def __init__(self, model, options, comm):
        self.model = model
        self.options = options
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()
        self.mesh = self.model.problem.Vh[PARAMETER].mesh()

        self.add_rule = options["add_rule"]
        self.add_number = options["add_number"]
        self.type_parameter = options["type_parameter"]

        self.particles = None

        self.number_particles = options["number_particles"]
        self.number_particles_old = self.number_particles
        self.number_particles_add = options["number_particles_add"]
        self.number_particles_all = self.number_particles + self.number_particles_add
        self.number_particles_all_old = self.number_particles_all

        if self.options["seed"] is 'random':
            seed = self.rank + int(np.random.rand(1)*100000)
        else:
            seed = self.rank + 2
        self.parRandom = Random(seed=seed)
        particles = []
        noise = dl.Vector()
        self.model.prior.init_vector(noise, "noise")
        for n in range(self.number_particles):
            particle = self.model.generate_vector(PARAMETER)
            self.parRandom.normal(1., noise)
            self.model.prior.sample(noise, particle, add_mean=True)
            particles.append(particle)
        self.particles = particles

        if self.number_particles_add > 0:
            for n in range(self.number_particles_add):
                particle = self.model.generate_vector(PARAMETER)
                self.parRandom.normal(1., noise)
                self.model.prior.sample(noise, particle, add_mean=True)
                self.particles.append(particle)

        self.particle_dimension = self.particles[0].get_local().size
        self.dimension = self.particle_dimension
        self.particles_gather = None

        self.time_communication = 0.
        self.time_computation = 0.

        self.communication()

        if self.options["is_projection"]:
            self.bases = None
            self.coefficient_dimension = self.options["coefficient_dimension"]
            if self.coefficient_dimension > self.options["rank_Hessian_average"]:
                self.coefficient_dimension = self.options["rank_Hessian_average"]
            self.dimension = self.coefficient_dimension
            self.Vh_COEFF = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.coefficient_dimension)
            self.particles_projection = [self.model.generate_vector(PARAMETER) for n in range(self.number_particles_all)]
            self.particles_complement = [self.model.generate_vector(PARAMETER) for n in range(self.number_particles_all)]
            self.coefficients = [self.generate_vector() for n in range(self.number_particles_all)]

            self.coefficients_gather = None

            self.communication_projection()

        self.particles_array = None
        self.particles_gather_array = None

        self.communication()

        # record the true mean and variance
        self.mean_posterior = None
        self.variance_posterior = None

    def generate_vector(self):
        if self.options["is_projection"]:
            return dl.Function(self.Vh_COEFF).vector()  # coefficient vector space
        else:
            return self.model.generate_vector(PARAMETER)

    def communication(self):  # gather particles from each processor to all processors
        time_communication = time.time()

        particles_array = np.empty([self.number_particles_all, self.particle_dimension], dtype=float)
        particles_gather_array = np.empty([self.nproc, self.number_particles_all, self.particle_dimension], dtype=float)
        for n in range(self.number_particles_all):
            particles_array[n, :] = self.particles[n].get_local()
        self.comm.Allgather(particles_array, particles_gather_array)

        self.particles_array = particles_array
        self.particles_gather_array = particles_gather_array

        self.particles_gather = [[None for n in range(self.number_particles_all)] for p in  range(self.nproc)]
        for p in range(self.nproc):
            for n in range(self.number_particles_all):
                self.particles_gather[p][n] = self.model.generate_vector(PARAMETER)
                self.particles_gather[p][n].set_local(particles_gather_array[p, n, :])

        self.time_communication += time.time() - time_communication

    def communication_projection(self):  # gather coefficients from each processor to all processors
        time_communication = time.time()

        coefficients_array = np.empty([self.number_particles_all, self.coefficient_dimension], dtype=float)
        coefficients_gather_array = np.empty([self.nproc, self.number_particles_all, self.coefficient_dimension], dtype=float)
        for n in range(self.number_particles_all):
            coefficients_array[n, :] = self.coefficients[n].get_local()
        self.comm.Allgather(coefficients_array, coefficients_gather_array)

        self.coefficients_gather = [[None for n in range(self.number_particles_all)] for p in  range(self.nproc)]
        for p in range(self.nproc):
            for n in range(self.number_particles_all):
                self.coefficients_gather[p][n] = self.generate_vector()
                self.coefficients_gather[p][n].set_local(coefficients_gather_array[p, n, :])

        self.time_communication += time.time() - time_communication

    def update_bases(self, U):
        self.bases = U
        if self.coefficient_dimension > U.nvec():
            self.coefficient_dimension = U.nvec()
            self.dimension = self.coefficient_dimension
            self.Vh_COEFF = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.coefficient_dimension)
            self.coefficients = [self.generate_vector() for n in range(self.number_particles_all)]

        self.projection()

    def update_dimension(self, d_average):
        self.coefficient_dimension = np.argmax(np.abs(d_average) <= self.options["tol_projection"])
        if self.coefficient_dimension == 0:
            self.coefficient_dimension = len(d_average)
        # if self.options["add_dimension"]:
        #     self.coefficient_dimension += self.options["add_dimension"]
        # if self.coefficient_dimension > len(d_average):
        #     self.coefficient_dimension = len(d_average)
        self.dimension = self.coefficient_dimension
        self.Vh_COEFF = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.coefficient_dimension)
        self.coefficients = [self.generate_vector() for n in range(self.number_particles_all)]

        self.projection()

    def projection(self):
        time_computation = time.time()

        for n in range(self.number_particles_all):
            self.particles_projection[n].zero()
            coefficients = np.empty(self.coefficient_dimension, dtype=float)
            particle = self.model.generate_vector(PARAMETER)
            particle.axpy(1.0, self.particles[n])
            particle.axpy(-1.0, self.model.prior.mean)
            phelp = self.model.generate_vector(PARAMETER)
            for r in range(self.coefficient_dimension):
                # coefficients[r] = self.model.prior.R.inner(particle, self.bases[r])
                self.model.prior.R.mult(particle, phelp)
                coefficients[r] = phelp.inner(self.bases[r])

                self.particles_projection[n].axpy(coefficients[r], self.bases[r])
            self.particles_projection[n].axpy(1.0, self.model.prior.mean)
            # print("coefficients", coefficients)
            self.coefficients[n].set_local(coefficients)

            self.particles_complement[n].zero()
            self.particles_complement[n].axpy(1.0, self.particles[n])
            self.particles_complement[n].axpy(-1.0, self.particles_projection[n])

        self.time_computation += time.time() - time_computation

        self.communication_projection()

    def prolongation(self):
        time_computation = time.time()

        for n in range(self.number_particles_all):
            self.particles_projection[n].zero()
            coefficients = self.coefficients[n].get_local()
            for r in range(self.coefficient_dimension):
                self.particles_projection[n].axpy(coefficients[r], self.bases[r])
            self.particles_projection[n].axpy(1.0, self.model.prior.mean)
            self.particles[n].zero()
            self.particles[n].axpy(1.0, self.particles_projection[n])
            self.particles[n].axpy(1.0, self.particles_complement[n])

        self.time_computation += time.time() - time_computation

        self.communication()

    def move(self, alpha, pstep):
        if self.options["is_projection"]:
            for m in range(self.number_particles_all):
                self.coefficients[m].axpy(alpha[m], pstep[m])
            self.communication_projection()
            self.prolongation()
        else:
            for m in range(self.number_particles_all):
                self.particles[m].axpy(alpha[m], pstep[m])
            self.communication()

    def add(self, variation):
        # add new particles by Laplace distribution at each of the particle for the transport map construction
        if self.add_rule:
            index = self.number_particles_all - self.number_particles_add
            for n in range(self.number_particles):
                sampler = GaussianLRPosterior(self.model.prior, variation.d[n], variation.U[n], mean=self.particles[n])
                noise = dl.Vector()
                self.model.prior.init_vector(noise, "noise")
                for m in range(self.add_number):
                    s_prior = self.model.generate_vector(PARAMETER)
                    s_posterior = self.model.generate_vector(PARAMETER)
                    self.parRandom.normal(1., noise)
                    self.model.prior.sample(noise, s_prior, add_mean=False)
                    sampler.sample(s_prior, s_posterior, add_mean=True)
                    self.particles.insert(index, s_posterior)
                    if self.options["is_projection"]:
                        self.particles_projection.insert(index, self.model.generate_vector(PARAMETER))
                        self.particles_complement.insert(index, self.model.generate_vector(PARAMETER))
                        self.coefficients.insert(index, self.generate_vector())
                        self.projection()
                    index += 1

            # update the number of particles used for constructing the transport map
            if self.add_rule == 1:  # add all the new particles for construction
                self.number_particles_old = self.number_particles
                self.number_particles += self.number_particles_old * self.add_number
            elif self.add_rule == 2:  # only add the ones added in the previous step
                self.number_particles_old = self.number_particles
                self.number_particles = self.number_particles_all - self.number_particles_add
            else:
                pass

            # update the total number of particles
            self.number_particles_all_old = self.number_particles_all
            self.number_particles_all += self.number_particles_old * self.add_number

        self.communication()
        if self.options["is_projection"]:
            self.communication_projection()

    def mean(self):
        # compute the mean of the particles
        mean = self.model.generate_vector(PARAMETER)
        for p in range(self.nproc):
            for m in range(self.number_particles_all):
                mean.axpy(1.0/(self.nproc*self.number_particles_all), self.particles_gather[p][m])

        mhelp = self.model.generate_vector(PARAMETER)
        self.model.prior.M.mult(mean, mhelp)
        meanL2norm = np.sqrt(mean.inner(mhelp))

        return mean, meanL2norm

    def trace(self):
        # compute the mean
        mean, _ = self.mean()
        # compute the trace of the sample covariance
        sample_trace = 0.
        if self.nproc*self.number_particles_all > 1:
            for p in range(self.nproc):
                for m in range(self.number_particles_all):
                    vhelp = self.model.generate_vector(PARAMETER)
                    vhelp.axpy(1.0, self.particles_gather[p][m])
                    vhelp.axpy(-1.0, mean)
                    sample_trace += np.sum(vhelp.get_local()**2)/(self.nproc*self.number_particles_all-1)

        return sample_trace

    def pointwise_variance(self):
        # compute the mean
        mean, _ = self.mean()
        # compute the pointwise variance
        variance = self.model.generate_vector(PARAMETER)
        varianceL2norm = 0.
        if self.nproc*self.number_particles_all > 1:
            for p in range(self.nproc):
                for m in range(self.number_particles_all):
                    vhelp = self.model.generate_vector(PARAMETER)
                    vhelp.axpy(1.0, self.particles_gather[p][m])
                    vhelp.axpy(-1.0, mean)
                    vsqured = vhelp.get_local()**2
                    vhelp.set_local(vsqured)
                    variance.axpy(1.0 / (self.nproc*self.number_particles_all-1), vhelp)

            vhelp = self.model.generate_vector(PARAMETER)
            self.model.prior.M.mult(variance, vhelp)
            varianceL2norm = np.sqrt(variance.inner(vhelp))

        return variance, varianceL2norm

    def statistics(self):
        # compute the mean of the particles
        mean = self.model.generate_vector(PARAMETER)
        for p in range(self.nproc):
            for m in range(self.number_particles_all):
                mean.axpy(1.0/(self.nproc*self.number_particles_all), self.particles_gather[p][m])

        mhelp = self.model.generate_vector(PARAMETER)
        self.model.prior.M.mult(mean, mhelp)
        meanL2norm = np.sqrt(mean.inner(mhelp))
        moment2 = self.model.generate_vector(PARAMETER)
        moment2.set_local(mean.get_local()**2)
        self.model.prior.M.mult(moment2, mhelp)
        moment2L2norm = np.sqrt(moment2.inner(mhelp))

        if self.mean_posterior is not None:
            mhelp2 = self.model.generate_vector(PARAMETER)
            mhelp2.axpy(1.0, self.mean_posterior)
            mhelp2.axpy(-1.0, mean)
            self.model.prior.M.mult(mhelp2, mhelp)
            meanErrorL2norm = np.sqrt(mhelp2.inner(mhelp))

        # compute the pointwise variance and trace of covariance
        variance = self.model.generate_vector(PARAMETER)
        varianceL2norm = 0.
        if self.nproc*self.number_particles_all > 1:
            for p in range(self.nproc):
                for m in range(self.number_particles_all):
                    vhelp = self.model.generate_vector(PARAMETER)
                    vhelp.axpy(1.0, self.particles_gather[p][m])
                    vhelp.axpy(-1.0, mean)
                    vsqured = vhelp.get_local()**2
                    vhelp.set_local(vsqured)
                    variance.axpy(1.0 / (self.nproc*self.number_particles_all-1), vhelp)

            vhelp = self.model.generate_vector(PARAMETER)
            self.model.prior.M.mult(variance, vhelp)
            varianceL2norm = np.sqrt(variance.inner(vhelp))

        if self.variance_posterior is not None:
            vhelp = self.model.generate_vector(PARAMETER)
            vhelp2 = self.model.generate_vector(PARAMETER)
            vhelp2.axpy(1.0, self.variance_posterior)
            vhelp2.axpy(-1.0, variance)
            self.model.prior.M.mult(vhelp2, vhelp)
            varianceErrorL2norm = np.sqrt(vhelp2.inner(vhelp))

        sample_trace = np.sum(variance.get_local())
        if self.variance_posterior is None:
            return mean, meanL2norm, moment2L2norm, variance, varianceL2norm, sample_trace
        else:
            return mean, meanL2norm, moment2L2norm, variance, varianceL2norm, sample_trace, meanErrorL2norm, varianceErrorL2norm

    def save(self, save_number=1, it=0):
        # save samples for visualization
        if save_number > self.number_particles_all:
            save_number = self.number_particles_all

        if False:
        # if self.rank == 0:
            mean = self.model.generate_vector(PARAMETER)
            for p in range(self.nproc):
                for m in range(self.number_particles_all):
                    mean.axpy(1.0/(self.nproc*self.number_particles_all), self.particles_gather[p][m])

            particle_fun = dl.Function(self.model.problem.Vh[PARAMETER], name='particle')
            particle_fun.vector().axpy(1.0, mean)
            filename = 'data/mean' + '_iteration_' + str(it) + '_isProjection_' + str(self.options["is_projection"]) + '_' + str(self.options["low_rank_Hessian"]) + '.xdmf'
            if dlversion() <= (1, 6, 0):
                dl.File(self.model.prior.R.mpi_comm(), filename) << particle_fun
            else:
                xf = dl.XDMFFile(self.model.prior.R.mpi_comm(), filename)
                xf.write(particle_fun)

            variance = self.model.generate_vector(PARAMETER)
            if self.nproc*self.number_particles_all > 1:
                for p in range(self.nproc):
                    for m in range(self.number_particles_all):
                        vhelp = self.model.generate_vector(PARAMETER)
                        vhelp.axpy(1.0, self.particles_gather[p][m])
                        vhelp.axpy(-1.0, mean)
                        vsqured = vhelp.get_local()**2
                        vhelp.set_local(vsqured)
                        variance.axpy(1.0 / (self.nproc*self.number_particles_all-1), vhelp)

            particle_fun = dl.Function(self.model.problem.Vh[PARAMETER], name='particle')
            particle_fun.vector().axpy(1.0, variance)
            filename = 'data/variance' + '_iteration_' + str(it) + '_isProjection_' + str(self.options["is_projection"]) + '_'  + str(self.options["low_rank_Hessian"]) + '.xdmf'
            if dlversion() <= (1, 6, 0):
                dl.File(self.model.prior.R.mpi_comm(), filename) << particle_fun
            else:
                xf = dl.XDMFFile(self.model.prior.R.mpi_comm(), filename)
                xf.write(particle_fun)

            for n in range(save_number):
                if self.type_parameter is 'field':
                    particle_fun = dl.Function(self.model.problem.Vh[PARAMETER], name='particle')
                    particle_fun.vector().axpy(1.0, self.particles[n])
                    filename = 'data/rank_' + str(self.rank) + '_particle_' + str(n) + '_iteration_' + str(it) + str(self.options["is_projection"]) + '_' + str(self.options["low_rank_Hessian"]) + '.xdmf'
                    if dlversion() <= (1, 6, 0):
                        dl.File(self.model.prior.R.mpi_comm(), filename) << particle_fun
                    else:
                        xf = dl.XDMFFile(self.model.prior.R.mpi_comm(), filename)
                        xf.write(particle_fun)
                elif self.type_parameter is 'vector':
                    filename = 'data/rank_' + str(self.rank) + '_particle_' + str(n) + '_iteration_' + str(it)
                    np.savez(filename, particle=self.particles[n].get_local())

    def plot_particles(self, particle, it):
        sample = np.zeros((2, particle.number_particles_all))
        for m in range(particle.number_particles_all):
            sample[:, m] = particle.particles[m].get_local()[:2]

        fig = plt.figure()
        plt.plot(sample[0, :], sample[1, :], 'r.')
        plt.axis('square')
        plt.xlim(-3., 3.)
        plt.ylim(-3., 3.)
        filename = 'figure/rank_' + str(self.rank) + '_samples_' + str(it) + '.pdf'
        fig.savefig(filename, format='pdf')
        plt.close()