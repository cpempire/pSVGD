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

import sys
import os

sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "../../../"))
from hippylib import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


class QoI:

    def __init__(self, Vh):
        test = dl.TestFunction(Vh[STATE])
        f = dl.Constant(1.0)
        self.f = dl.assemble(f*test*dl.dx)

    def eval(self, x):
        return self.f.inner(x[STATE])


def u_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS )  # or x[0] > 1.0 - dl.DOLFIN_EPS


def pde_varf(u, m, p):
    return dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx + u * p * dl.dx - m * p * dl.dx


# create mesh
dl.set_log_active(False)
sep = "\n" + "#" * 80 + "\n"
nx = 16
mesh = dl.IntervalMesh(dl.mpi_comm_self(), nx, 0., 1.)

# define function space
Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh2, Vh1, Vh2]

ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
if rank == 0:
    print(sep, "Set up the mesh and finite element spaces", sep)
    print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))


# define boundary conditions and PDE
u_bdr = dl.Expression("1-x[0]", degree=1)
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

qoi = QoI(Vh)

# define the misfit
ntargets = 15
targets = np.array(range(ntargets))/(ntargets+1) + 1./(ntargets+1)

if rank == 0:
    print("Number of observation points: {0}".format(ntargets))
misfit = PointwiseStateObservation(Vh[STATE], targets)

# define the prior
gamma = 0.1
delta = 1.0
prior = LaplacianPrior(Vh[PARAMETER], gamma, delta)

if rank == 0:
    print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta,
                                                                                                            gamma,
                                                                                                            2))
# Generate synthetic observations
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
rel_noise = 0.01
MAX = misfit.d.norm("linf")
noise_std_dev = rel_noise * MAX
parRandom.normal_perturb(noise_std_dev, misfit.d)
if rank == 0:
    d_array = misfit.d.get_local()
else:
    d_array = None
d_array = comm.bcast(d_array, root=0)
misfit.d.set_local(d_array)

misfit.noise_variance = noise_std_dev * noise_std_dev


# define the model
model = Model(pde, prior, misfit, qoi)

