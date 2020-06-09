
from __future__ import absolute_import, division, print_function

from ..modeling.variables import PARAMETER


class MetricPrior:
    # define the metric M for the kernel k(pn, pm) = exp(-(pn-pm)^T * M * (pn - pm))
    def __init__(self, model, particle, variation, options):
        self.model = model  # forward model
        self.particle = particle
        self.variation = variation
        self.options = options

    def mult(self, phat, pout, n):

        pout.zero()

        if self.options["is_projection"]:
            # covariance/regularization
            pout.axpy(1.0, phat)
        else:
            # covariance/regularization
            phelp = self.model.generate_vector(PARAMETER)
            self.model.prior.R.mult(phat, phelp)
            pout.axpy(1.0, phelp)

    def inner(self, phat1, phat2, n):

        phelp = self.model.generate_vector(PARAMETER)
        self.mult(phat1, phelp, n)

        return phat2.inner(phelp)

    def update(self, particle, variation):
        self.particle = particle
        self.variation = variation


class MetricPosteriorSeparate:
    # define the metric M for the kernel k(pn, pm) = exp(-(pn-pm)^T * M * (pn - pm))
    def __init__(self, model, particle, variation, options):
        self.model = model  # forward model
        self.particle = particle
        self.variation = variation
        self.options = options

    def mult(self, phat, pout, n):

        pout.zero()

        if self.options["is_projection"]:
            self.variation.hessian(phat, pout, n)
            pout.axpy(1.0, phat)
        else:
            # covariance/regularization
            phelp = self.model.generate_vector(PARAMETER)
            self.model.prior.R.mult(phat, phelp)
            pout.axpy(1.0, phelp)
            # negative log likelihood function
            phelp1 = self.model.generate_vector(PARAMETER)
            self.variation.hessian(phelp, phelp1, n)
            self.model.prior.R.mult(phelp1, phelp)

            pout.axpy(1.0, phelp)

    def inner(self, phat1, phat2, n):

        phelp = self.model.generate_vector(PARAMETER)
        self.mult(phat1, phelp, n)

        return phat2.inner(phelp)

    def update(self, particle, variation):
        self.particle = particle
        self.variation = variation


class MetricPosteriorAverage:
    # define the metric M for the kernel k(pn, pm) = exp(-(pn-pm)^T * M * (pn - pm))
    def __init__(self, model, particle, variation, options):
        self.model = model  # forward model
        self.particle = particle
        self.variation = variation
        self.options = options

    def mult(self, phat, pout, n):

        pout.zero()

        if self.options["is_projection"]:
            self.variation.hessian_average(phat, pout)
            pout.axpy(1.0, phat)
        else:
            # covariance/regularization
            phelp = self.model.generate_vector(PARAMETER)
            self.model.prior.R.mult(phat, phelp)
            pout.axpy(1.0, phelp)
            # negative log likelihood function
            phelp1 = self.model.generate_vector(PARAMETER)
            self.variation.hessian_average(phelp, phelp1)
            self.model.prior.R.mult(phelp1, phelp)
            pout.axpy(1.0, phelp)

    def inner(self, phat1, phat2, n):

        phelp = self.model.generate_vector(PARAMETER)
        self.mult(phat1, phelp, n)

        return phat2.inner(phelp)

    def update(self, particle, variation):
        self.particle = particle
        self.variation = variation
