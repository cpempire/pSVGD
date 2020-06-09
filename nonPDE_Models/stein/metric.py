
from __future__ import absolute_import, division, print_function
import numpy as np


class MetricPrior:
    # define the metric M for the kernel k(pn, pm) = exp(-(pn-pm)^T * M * (pn - pm))
    def __init__(self, model, particle, variation, options):
        self.model = model  # forward model
        self.particle = particle
        self.variation = variation
        self.options = options

        if self.options["kernel_vectorized"]:
            if self.options["is_projection"]:
                self.matrix = np.eye(particle.coefficient_dimension)
            else:
                self.matrix = self.model.prior.CovInv

    def mult(self, phat, pout, n):

        pout[:] = 0.

        if self.options["is_projection"]:
            # covariance/regularization
            pout += 1.0 * phat
        else:
            # covariance/regularization
            phelp = self.model.generate_vector()
            self.model.prior.R.mult(phat, phelp)
            pout += 1.0 * phelp

    def inner(self, phat1, phat2, n):

        phelp = self.model.generate_vector()
        self.mult(phat1, phelp, n)

        return phat2.dot(phelp)

    def update(self, particle, variation):
        self.particle = particle
        self.variation = variation

        if self.options["kernel_vectorized"]:
            if self.options["is_projection"]:
                self.matrix = np.eye(particle.coefficient_dimension)
            else:
                self.matrix = self.model.prior.CovInv


class MetricPosteriorAverage:
    # define the metric M for the kernel k(pn, pm) = exp(-(pn-pm)^T * M * (pn - pm))
    def __init__(self, model, particle, variation, options):
        self.model = model  # forward model
        self.particle = particle
        self.variation = variation
        self.options = options
        if self.options["kernel_vectorized"]:
            H = self.variation.hessian_average_matrix()
            if self.options["is_projection"]:
                R = np.eye(H.shape[0])
            else:
                R = self.model.prior.CovInv

            self.matrix = R + R.dot(H).dot(R)

    def mult(self, phat, pout, n):

        pout[:] = 0.

        if self.options["is_projection"]:
            self.variation.hessian_average(phat, pout)
            pout += 1.0 * phat
        else:
            # covariance/regularization
            phelp = self.model.generate_vector()
            self.model.prior.R.mult(phat, phelp)
            pout += 1.0 * phelp
            # negative log likelihood function
            phelp1 = self.model.generate_vector()
            self.variation.hessian_average(phelp, phelp1)
            self.model.prior.R.mult(phelp1, phelp)
            pout += 1.0 * phelp

    def inner(self, phat1, phat2, n):

        phelp = self.model.generate_vector()
        self.mult(phat1, phelp, n)

        return phat2.dot(phelp)

    def update(self, particle, variation):
        self.particle = particle
        self.variation = variation
        if self.options["kernel_vectorized"]:
            H = self.variation.hessian_average_matrix()
            if self.options["is_projection"]:
                R = np.eye(H.shape[0])
            else:
                R = self.model.prior.CovInv

            self.matrix = R + R.dot(H).dot(R)


class MetricPosteriorSeparate:
    # define the metric M for the kernel k(pn, pm) = exp(-(pn-pm)^T * M * (pn - pm))
    def __init__(self, model, particle, variation, options):
        self.model = model  # forward model
        self.particle = particle
        self.variation = variation
        self.options = options

    def mult(self, phat, pout, n):

        pout[:] = 0.

        if self.options["is_projection"]:
            self.variation.hessian(phat, pout, n)
            pout += 1.0 * phat
        else:
            # covariance/regularization
            phelp = self.model.generate_vector()
            self.model.prior.R.mult(phat, phelp)
            pout += 1.0 * phelp
            # negative log likelihood function
            phelp1 = self.model.generate_vector()
            self.variation.hessian(phelp, phelp1, n)
            self.model.prior.R.mult(phelp1, phelp)

            pout += 1.0 * phelp

    def inner(self, phat1, phat2, n):

        phelp = self.model.generate_vector()
        self.mult(phat1, phelp, n)

        return phat2.dot(phelp)

    def update(self, particle, variation):
        self.particle = particle
        self.variation = variation



