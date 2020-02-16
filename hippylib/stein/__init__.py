# contact: Peng Chen, peng@ices.utexas.edu,
# written on Novemeber 19, 2018
# updated for parallel version on Jan 2, 2019
# updated for projection version on Jan 10, 2019
# updated for fisher version on Jan 31, 2019


from __future__ import absolute_import, division, print_function

from .metric import MetricPrior, MetricPosteriorSeparate, MetricPosteriorAverage
from .kernel import Kernel
from .variation import Variation
from .newtonSeparated import NewtonSeparated
from .newtonCoupled import NewtonCoupled
from .gradientDescent import GradientDescent
from .options import options
from .particle import Particle
