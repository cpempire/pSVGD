from __future__ import absolute_import, division, print_function

import numpy as np
import math


class Model:
    """
    This class contains the full description of the cost function, its gradient, and Hessian action
    """
    
    def __init__(self, prior, misfit, qoi=None):
        """
        Create a model given:
            - prior: the prior component of the cost functional
        """
        self.prior = prior
        self.misfit = misfit
        self.qoi = qoi

    def generate_vector(self):

        x = np.zeros(self.prior.dimension)
        
        return x
                
    def cost(self, x):
        """
        Evaluate the cost function.
        """
        misfit_cost = self.misfit.cost(x)

        reg_cost = self.prior.cost(x)

        return [misfit_cost+reg_cost, reg_cost, misfit_cost]

    def gradient(self, x, mg, misfit_only=False):
        """
        Evaluate the gradient.        
        Returns the norm of the gradient in the correct inner product :math:`g_norm = sqrt(g,g)`
        """ 
        tmp = self.generate_vector()
        self.misfit.grad(x, tmp)
        mg += tmp
        if not misfit_only:
            self.prior.grad(x, tmp)
            mg += tmp
        
        self.prior.Msolver.solve(tmp, mg)
        return math.sqrt(mg.dot(tmp))

    def hessian_vector_product(self, x, dir, mh):
        """
        Evaluate the Hessian at x acting in direction dir, and return it to mh
        """
        # todo implement the Hessian action at x in direction dir and return it to mh

        return 1