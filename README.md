Projected Stein variational methods for high-dimensional Bayesian inference 

The nonPDE models are implemented in nonPDE_Models, using HIPS/autograd to compute gradients

The PDE model example is implemented in PDE_Models, using FEniCS as backend solver

The main source files are in hippylib/stein/

The application files are in applications/

The main features of this implementation include

(1) A Stein variational gradient decent method (SVGD);

(2) A Stein variational Newton method (SVN);

(3) Different methods for kernel construction;

(4) Projected SVGD and projected SVN;

(5) Parallel computation using MPI;

(6) Dynamic sampling and optimization;

(7) Work for both parameter field and finite-dimensional parameters.
