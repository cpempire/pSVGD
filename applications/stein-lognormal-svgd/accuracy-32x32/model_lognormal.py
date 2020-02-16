# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from __future__ import absolute_import, division, print_function

import dolfin as dl
import numpy as np
import pickle

import sys
import os

sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "../../../"))
from hippylib import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


class FluxQOI(object):
    def __init__(self, Vh, dsGamma):
        self.Vh = Vh
        self.dsGamma = dsGamma
        self.n = dl.Constant((0., 1.))  # dl.FacetNormal(Vh[STATE].mesh())

        self.u = None
        self.m = None
        self.L = {}

    def form(self, x):
        # return dl.avg(dl.exp(x[PARAMETER])*dl.dot( dl.grad(x[STATE]), self.n) )*self.dsGamma
        return dl.exp(x[PARAMETER]) * dl.dot(dl.grad(x[STATE]), self.n) * self.dsGamma

    def eval(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter a are accessed.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        return dl.assemble(self.form([u, m]))


class QoI:

    def __init__(self, Vh):
        test = dl.TestFunction(Vh[STATE])
        f = dl.Constant(1.0)
        self.f = dl.assemble(f*test*dl.dx)

    def eval(self, x):
        return self.f.inner(x[STATE])


class GammaBottom(dl.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[1]) < dl.DOLFIN_EPS)


def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


def v_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)


def true_model(Vh, gamma, delta, anis_diff):
    prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff)
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue


# define the PDE
def pde_varf(u, m, p):
    return dl.exp(m) * dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx - dl.Constant(0.0) * p * dl.dx


dl.set_log_active(False)
sep = "\n" + "#" * 80 + "\n"
ndim = 2
nx = 32
ny = 32
mesh = dl.UnitSquareMesh(dl.mpi_comm_self(), nx, ny)

Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh2, Vh1, Vh2]

ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
if rank == 0:
    print(sep, "Set up the mesh and finite element spaces", sep)
    print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))

u_bdr = dl.Expression("x[1]", degree=1)
u_bdr0 = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)


pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
# if dlversion() <= (1, 6, 0):
#     pde.solver = dl.PETScKrylovSolver("cg", amg_method())
#     pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", amg_method())
#     pde.solver_adj_inc = dl.PETScKrylovSolver("cg", amg_method())
# else:
#     pde.solver = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
#     pde.solver_fwd_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
#     pde.solver_adj_inc = dl.PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
# pde.solver.parameters["relative_tolerance"] = 1e-15
# pde.solver.parameters["absolute_tolerance"] = 1e-20
# pde.solver_fwd_inc.parameters = pde.solver.parameters
# pde.solver_adj_inc.parameters = pde.solver.parameters


# define the QOI
GC = GammaBottom()
marker = dl.FacetFunction("size_t", mesh)
marker.set_all(0)
GC.mark(marker, 1)
dss = dl.Measure("ds", subdomain_data=marker)
qoi = FluxQOI(Vh, dss(1))

# qoi = QoI(Vh)

# define the misfit
n1d = 7
ntargets = n1d**2
targets = np.zeros((ntargets, 2))
x1d = np.array(range(n1d))/float(n1d+1) + 1/float(n1d+1)
for i in range(n1d):
    for j in range(n1d):
        targets[i*n1d+j, 0] = x1d[i]
        targets[i*n1d+j, 1] = x1d[j]
if rank == 0:
    print("Number of observation points: {0}".format(ntargets))
misfit = PointwiseStateObservation(Vh[STATE], targets)


# define the prior
gamma = 0.1
delta = 1.0
prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta)

if rank == 0:
    print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta,
                                                                                                            gamma,
                                                                                                            2))
# Generate synthetic observations
try:
    print("load observation data")
    data = pickle.load(open("data/observation_data.p","rb"))
    d_array = data["d_array"]
    noise_std_dev = data["noise_std_dev"]
except:
    print("create observation data")
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    parRandom.normal(1., noise)
    # use the same noise across processors
    if rank == 0:
        noise_array = noise.get_local()
    else:
        noise_array = None
    noise_array = comm.bcast(noise_array, root=0)
    noise.set_local(noise_array)

    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)

    if rank == 0:
        filename = 'data/particle_true.xdmf'
        particle_fun = dl.Function(Vh[PARAMETER], name='particle')
        particle_fun.vector().axpy(1.0, mtrue)
        if dlversion() <= (1, 6, 0):
            dl.File(mesh.mpi_comm(), filename) << particle_fun
        else:
            xf = dl.XDMFFile(mesh.mpi_comm(), filename)
            xf.write(particle_fun)

    utrue = pde.generate_state()
    x = [utrue, mtrue, None]
    pde.solveFwd(x[STATE], x, 1e-9)
    misfit.B.mult(x[STATE], misfit.d)
    rel_noise = 0.05
    MAX = misfit.d.norm("linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev, misfit.d)
    if rank == 0:
        d_array = misfit.d.get_local()
    else:
        d_array = None
    d_array = comm.bcast(d_array, root=0)
    data = dict()
    data["d_array"] = d_array
    data["noise_std_dev"] = noise_std_dev
    pickle.dump(data, open("data/observation_data.p", "wb"))

misfit.d.set_local(d_array)
misfit.noise_variance = noise_std_dev * noise_std_dev

# define the model
model = Model(pde, prior, misfit, qoi)


