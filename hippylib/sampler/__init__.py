"""
Import MCMC samplers
Shiwei Lan @ U of Warwick, 2016
Created July 26, 2016
"""

# geometric infinite-dimensional MCMC's

# from geoinfMC import geoinfMC
#
# # geoinfMC using hippylib (FEniCS 1.6.0/1.5.0)
# try:
#     from geoinfMC_hippy import geoinfMC
# except Exception:
#     pass

# Dimension-independent likelihood-informed (DILI) MCMC by Tiangang Cui, Kody J.H. Law, and Youssef M. Marzouk

# from DILI import DILI
#
# # DILI that uses dolfin
# try:
#     from DILI_dolfin import DILI
# except ImportError:
#     pass

from .DILI_hippy import DILI
from .geometry import Geometry
from .whiten import wht_prior, wht_Hessian
from .randomizedEigensolver_ext import singlePassG_prec, singlePassGx