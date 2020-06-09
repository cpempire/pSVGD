
from __future__ import absolute_import, division, print_function

from .metric import MetricPrior, MetricPosteriorSeparate, MetricPosteriorAverage
from .kernel import Kernel
from .variation import Variation
# from .newtonSeparated import NewtonSeparated
# from .newtonCoupled import NewtonCoupled
from .gradientDescent import GradientDescent
from .options import options
from .particle import Particle
from .model import Model
from .prior import FiniteDimensionalPrior, BrownianMotion, Laplacian



# Error: Cannot install open-mpi because conflicting formulae are installed.
#   mpich: because both install MPI compiler wrappers
#
# Please `brew unlink mpich` before continuing.
#
# Unlinking removes a formula's symlinks from /usr/local. You can
# link the formula again after the install finishes. You can --force this
# install, but the build may fail or cause obscure side effects in the
# resulting software.
# peng@MacBook-Pro toy2d % brew unlink mpich
# Unlinking /usr/local/Cellar/mpich/3.3.1... 460 symlinks removed
