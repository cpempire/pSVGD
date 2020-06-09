import autograd.numpy as np
from autograd import grad, jacobian, hessian_vector_product
from autograd.numpy import multiply as ewm
from autograd.misc import flatten

from mpi4py import MPI
comm = MPI.COMM_WORLD

import sys
import os
import pickle
import time

path = "../../"
sys.path.append(path)
from stein import *
from stein.randomizedEigensolver import doublePass

import matplotlib.pyplot as plt
from utils.integration import Euler, RK2, RK4
from utils.interpolation import piecewiseConstant, piecewiseLinear, cubicSpline


class hvp():
    def __init__(self, hx):

        self.x = None
        self.hx = hx

    def update_x(self, x):

        self.x = x

    def mult(self, xhat, out):

        out[:] = self.hx(self.x, xhat)


class Misfit():

    def __init__(self, configurations, parameters, controls, simulation_first_confirmed):

        self.configurations = configurations
        self.parameters = parameters
        self.controls = controls
        self.simulation_first_confirmed = simulation_first_confirmed

        flat_args, self.unflatten = flatten(self.controls)

        # self.obs = obs
        # self.loc = loc
        # if noise_covariance is not None:
        #     self.Gamma_noise_inv = np.linalg.inv(noise_covariance)

        self.gx = grad(self.cost)
        self.J = jacobian(self.forward)
        self.hx = hessian_vector_product(self.cost)
        self.hvp = hvp(self.hx)

        y0, t_total, N_total, number_group, population_proportion, \
        t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

        self.N_total = N_total
        self.number_group = number_group
        self.t_control = t_control
        self.dimension = len(self.t_control)
        self.number_time_dependent_controls = number_time_dependent_controls
        self.y0 = y0
        self.t_total = t_total

        self.interpolation = piecewiseLinear

        if number_group > 1:
            # contact matrix
            school_closure = True

            # calendar from February 15th
            weekday = [2, 3, 4, 5, 6]
            # calendar from April 1st
            # weekday = [0, 1, 2, 5, 6]
            # calendar from May 1st
            # weekday = [0, 3, 4, 5, 6]
            calendar = np.zeros(1000 + 1, dtype=int)
            # set work days as 1 and school days as 2
            for i in range(1001):
                if np.remainder(i, 7) in weekday:
                    calendar[i] = 1
                    if not school_closure:  # and i < 45
                        calendar[i] = 2
            self.calendar = calendar

            contact = np.load("utils/contact_matrix.npz")
            self.c_home = contact["home"]
            self.c_school = contact["school"]
            self.c_work = contact["work"]
            self.c_other = contact["other"]

            self.contact_full = self.c_home + 5. / 7 * ((1 - school_closure) * self.c_school + self.c_work) + self.c_other

        self.load_data(fips)
        # self.initialization()

    def load_data(self, fips):
        # load data
        states = pickle.load(open("utils/states_dictionary_moving_average", 'rb'))

        # population 0-4, 5-9, ..., 80-84, 85+
        population = states[fips]['population']
        N_total = np.sum(population)

        # print("fips = ", fips, "population = ", N_total)

        state_name = states[fips]['name']

        first_confirmed = np.where(states[fips]["positive"] > 100)[0][0]
        data_confirmed = states[fips]["positive"][first_confirmed:]
        self.number_days_data = len(data_confirmed)
        self.data_confirmed = data_confirmed

        first_hospitalized = np.where(states[fips]["hospitalizedCurrently"] > 10)[0][0]
        self.lag_hospitalized = first_hospitalized - first_confirmed
        self.data_hospitalized = states[fips]["hospitalizedCurrently"][first_hospitalized:]

        first_deceased = np.where(states[fips]["death"] > 10)[0][0]
        self.lag_deceased = first_deceased - first_confirmed
        self.data_deceased = states[fips]["death"][first_deceased:]

        # self.obs = np.append(np.log(self.data_deceased), np.log(np.diff(self.data_deceased)))
        self.obs = np.log(self.data_hospitalized)
        # self.Gamma_noise_inv = np.diag(1./np.power(0.01*self.obs, 2))
        self.Gamma_noise_inv = np.eye(len(self.obs))

        # day = self.simulation_first_confirmed + self.lag_deceased
        # self.loc = np.arange(day, day+len(self.data_deceased))
        day = self.simulation_first_confirmed + self.lag_hospitalized
        self.loc = np.arange(day, day+len(self.data_hospitalized))

    # def initialization(self):
    #     # initialize the optimization problem with controls terminate at the end of observation (today)
    #
    #     y0, t_total, N_total, number_group, population_proportion, \
    #     t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = self.configurations
    #
    #     alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = self.controls
    #
    #     t0 = time.time()
    #     solution = RK2(self.seir, self.y0, self.t_total, self.parameters, self.controls)
    #     # np.savez(savefilename, t=t, solution=solution, controls=controls)
    #     print("solve time by RK2 method", time.time() - t0)
    #
    #     solution_group = self.grouping(solution)
    #
    #     print("# total infected = ", self.N_total - solution_group[-1, 0],
    #           "# total death = ", solution_group[-1, 7],
    #           "maximum # hospitalized = ", np.max(solution_group[:, 5]))
    #
    #     self.simulation_first_confirmed = np.where(solution_group[:, 8] >= self.data_confirmed[0])[0][0]
    #     # simulation_first_confirmed = np.where(solution_group[:, 8] > data_confirmed[0]-1)[0][0]
    #     # simulation_first_confirmed = 40
    #     print("day for first 100 confirmed = ", self.simulation_first_confirmed)
    #
    #     # control frequency and time
    #     # number_days_per_control_change = 7
    #     day = self.simulation_first_confirmed + self.lag_deceased
    #     self.loc = np.arange(day, day+len(self.data_deceased))
    #
    #     number_days = self.simulation_first_confirmed + len(self.data_confirmed)
    #     number_control_change_times = number_days
    #     t_total = np.linspace(0, number_days, number_days + 1)
    #     t_control = np.linspace(0, number_days-1, number_days)
    #     self.t_total = t_total
    #     self.t_control = t_total
    #     self.dimension = len(self.t_control)
    #     alpha = 0.1 * np.ones(len(self.t_control))
    #     self.controls = (alpha, q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q)
    #
    #     flat_args, self.unflatten = flatten(self.controls)
    #
    #     self.configurations = (y0, t_total, N_total, number_group, population_proportion,
    #                   t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls)

    def contact_rate(self, t):

        if self.number_group == 1:
            return 1.
        else:
            # ###### definition of the contact rate

            # contact_full = np.block([ewm(contact_full, np.tile(1-high_risk_distribution, [number_age_group, 1])),
            #                          ewm(contact_full, np.tile(high_risk_distribution, [number_age_group, 1]))])
            # contact_full = np.tile(contact_full, [number_risk_group, 1])

            # contact_full = np.tile(contact_full, [2, 2])

            # contact_full = 5*np.ones((number_group, number_group))

            if self.calendar[np.int(np.floor(t))] == 2:
                contact = self.c_home + self.c_school + self.c_work + self.c_other
            elif self.calendar[np.int(np.floor(t))] == 1:
                contact = self.c_home + self.c_work + self.c_other
            else:
                contact = self.c_home + self.c_other
            # else:
            #     contact = c_home + 0.1 * (c_work + c_other)
            # if calendar[np.int(np.floor(t))] == 1:
            #     contact = c_home + 0.1*(c_work + c_other)
            # else:
            #     contact = c_home

            # # construct contact matrix by splitting each age group into two by low and high risk proportion
            # contact = np.block([ewm(contact, np.tile(1 - high_risk_distribution, [number_age_group, 1])),
            #                          ewm(contact, np.tile(high_risk_distribution, [number_age_group, 1]))])
            # contact = np.tile(contact, [number_risk_group, 1])

            contact = np.tile(contact, [2, 2])

            # constant contact
            # contact = 10*np.ones((number_group, number_group))

            return contact

    def proportion2factor(self, proportion, r1, r2):
        # define a function to transform proportion to factor, e.g., from prop_EI to tau with r1=sigma_I, r2 = sigma_A
        factor = np.divide(ewm(r2, proportion), r1 + ewm(r2 - r1, proportion))

        return factor

    def seir(self, y, t, parameters, controls, stochastic=False):
        # define the right hand side of the ode systems given state y, time t, parameters, and controls

        if self.number_group > 1:
            y = y.reshape((10, self.number_group))

        S, E, Q, A, I, H, R, D, Tc, Tu = y

        q, tau, HFR, kappa, beta, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = parameters
        # _, _, _, delta, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q = parameters
        alpha, = controls
        alpha = (np.tanh(alpha) + 1)/2

        alpha = self.interpolation(t, self.t_control, alpha)

        # alpha, q, tau, HFR, kappa = [self.interpolation(t, self.t_control, controls[i]) for i in range(self.number_time_dependent_controls)]

        # tau_p = self.interpolation(t - np.max(1./sigma), self.t_control, controls[2])
        # tau_p = self.interpolation(t, self.t_control, controls[2])
        # tau_p = tau

        # IHR = np.divide(kappa, np.max(kappa) + tau_p)

        IHR = kappa

        QHR = ewm(tau, IHR)

        # gamma_A = gamma_I

        pi = self.proportion2factor(IHR, eta_I, gamma_I)
        nu = self.proportion2factor(HFR, mu, gamma_H)
        rho = self.proportion2factor(QHR, eta_Q, gamma_Q)

        contact = self.contact_rate(t)

        # theta_I = 2 - tau
        # theta_A = 1 - tau

        theta_I = 1. - 0*tau
        theta_A = 1. - 0*tau
        delta = 1. + 0*delta

        C_E = ewm(1-alpha,
                  ewm(1-q, ewm(delta, np.dot(contact, ewm(theta_I, np.divide(I, self.N_total))))) +
                  np.dot(contact, ewm(theta_A, np.divide(A, self.N_total)))
                  )

        C_Q = ewm(1-alpha,
                  ewm(q, ewm(delta, np.dot(contact, ewm(theta_I, np.divide(I, self.N_total)))))
                  )

        if stochastic:
            zeros = np.zeros(np.size(S))
            S = np.max([zeros, S], axis=0)
            E = np.max([zeros, E], axis=0)
            Q = np.max([zeros, Q], axis=0)
            A = np.max([zeros, A], axis=0)
            I = np.max([zeros, I], axis=0)
            H = np.max([zeros, H], axis=0)

        P1 = ewm(beta, ewm(C_E, S))
        P2 = ewm(beta, ewm(C_Q, S))
        P3 = ewm(tau, ewm(sigma, E))
        P4 = ewm(1 - tau, ewm(sigma, E))
        P5 = ewm(rho, ewm(eta_Q, Q))
        P6 = ewm(1 - rho, ewm(gamma_Q, Q))
        P7 = ewm(gamma_A, A)
        P8 = ewm(pi, ewm(eta_I, I))
        P9 = ewm(1 - pi, ewm(gamma_I, I))
        P10 = ewm(nu, ewm(mu, H))
        P11 = ewm(1 - nu, ewm(gamma_H, H))

        if stochastic:
            P1 = np.random.poisson(P1)
            P2 = np.random.poisson(P2)
            P3 = np.random.poisson(P3)
            P4 = np.random.poisson(P4)
            P5 = np.random.poisson(P5)
            P6 = np.random.poisson(P6)
            P7 = np.random.poisson(P7)
            P8 = np.random.poisson(P8)
            P9 = np.random.poisson(P9)
            P10 = np.random.poisson(P10)
            P11 = np.random.poisson(P11)

        dS = - P1 - P2
        dE = P1 - P3 - P4
        dQ = P2 - P5 - P6
        dA = P4 - P7
        dI = P3 - P8 - P9
        dH = P8 + P5 - P10 - P11
        dR = P7 + P9 + P11 + P6
        dD = P10
        dTc = P3 + P2   # + quarantined, P2
        dTu = P4
        dydt = np.array([dS, dE, dQ, dA, dI, dH, dR, dD, dTc, dTu]).flatten("C")

        return dydt

    def grouping(self, solution):
        # grouping different groups in solution into one group

        if self.number_group > 1:
            # solution_help = np.zeros((solution.shape[0], 10))
            # for i in range(10):
            #     solution_help[:, i] = np.sum(solution[:, i * self.number_group:(i+1) * self.number_group], axis=1)
            # solution = solution_help

            solution_help = []
            for i in range(10):
                solution_help = np.append(solution_help, np.sum(solution[:, i * self.number_group:(i+1) * self.number_group], axis=1))
            solution = np.reshape(solution_help, (10, solution.shape[0])).T

        return solution

    # def solution(self, x):
    #
    #     dx = np.diff(x)
    #     u = np.zeros_like(x)
    #     for i in range(1, len(x)):
    #         u[i] = u[i - 1] + self.beta * u[i - 1] * (1 - u[i - 1] ** 2) / (1 + u[i - 1] ** 2) * self.dt[i - 1] + dx[i - 1]
    #
    #     return u

    def solution(self, x):

        controls = self.unflatten(x)
        solution = RK2(self.seir, self.y0, self.t_total, self.parameters, controls)
        solution = self.grouping(solution)

        return solution[:,5]

    def forward(self, x):

        controls = self.unflatten(x)
        solution = RK2(self.seir, self.y0, self.t_total, self.parameters, controls)
        solution = self.grouping(solution)

        # u = np.append(np.log(solution[self.loc, 7]), np.log(np.diff(solution[self.loc, 7])))
        u = np.log(solution[self.loc, 5])

        return u

    def cost(self, x):

        u = self.forward(x)
        diff = (self.obs-u)
        negative_log_likelihood = 0.5 * np.dot(np.dot(diff, self.Gamma_noise_inv), diff)

        controls = self.unflatten(x)

        # alpha, = controls
        # alpha = (np.tanh(alpha) + 1) / 2
        # penalty_controls = np.sum(np.mean(ewm(1., np.power(np.abs(np.diff(alpha, axis=0))-np.diff(alpha, axis=0), 2.)), axis=0))
        # penalty_controls = np.sum(np.mean(ewm(1., np.power(np.abs(np.diff(controls[0], axis=0))-np.diff(controls[0], axis=0), 1.)), axis=0))
        penalty_controls = 0.*np.sum(np.mean(ewm(1., np.power(np.abs(np.diff(controls[0], axis=0))-np.diff(controls[0], axis=0), 1.)), axis=0))
        penalty_controls += np.sum(np.mean(ewm(1., np.power(np.abs(np.diff(controls[0], axis=0)), 1.)), axis=0))

        # penalty_controls = np.sum(np.mean(ewm(1., np.power(np.abs(np.diff(controls[0], axis=0)), 2.)), axis=0))
        # penalty_controls = np.sum(np.mean(ewm(1., np.power(np.abs(np.diff(controls[0], axis=0)), 2.)), axis=0))

        # print("cost, penalty = ", negative_log_likelihood, penalty_controls)

        # # # # penalization on HFR, beta, delta, kappa, sigma, eta_I, eta_Q, mu, gamma_I, gamma_A, gamma_H, gamma_Q from mean
        # result = 0.
        # for j in range(0, len(parameters)):
        #     result = result + 0.5/500 * np.sum(np.power(np.divide(np.abs(controls[j+1] - parameters[j]), parameters[j]/2), 1))

        # penalty_controls += result

        # print("cost, penalty = ", negative_log_likelihood, 10.*penalty_controls)

        return negative_log_likelihood + 1.*penalty_controls
        # return negative_log_likelihood + 100.*result
        # return negative_log_likelihood + 0.*penalty_controls

    def grad(self, x, g):

        g[:] = self.gx(x)

        return g

    def post(self, x):
        prior = Laplacian(self.dimension)
        negative_log_likelihood = self.cost(x)
        post = np.exp(-np.dot(x, prior.CovInv).dot(x)/2. - negative_log_likelihood)

        return post

    def eigdecomp(self, x, k):

        # self.hvp.update_x(x)
        # d, U = doublePass(self.hvp, self.Omega, k, s=1, check=False)

        J = self.J(x)
        if len(np.shape(J)) == 1:
            J = np.array([J])
        Gauss_Newton = np.dot(np.dot(J.T, self.Gamma_noise_inv), J)
        d, U = np.linalg.eigh(Gauss_Newton)

        sort_perm = np.abs(d).argsort()

        sort_perm = sort_perm[::-1]
        d = d[sort_perm[:k]]
        U = U[:, sort_perm[:k]]

        return d, U

    def geom(self, x, geom_ord=[0], k=100):

        loglik = None
        agrad = None
        HessApply = None
        eigs = None

        # get log-likelihood
        if any(s >= 0 for s in geom_ord):
            loglik = - self.cost(x)

        # get gradient
        if any(s >= 1 for s in geom_ord):
            g = np.zeros_like(x)
            agrad = - self.grad(x, g)

        # get Hessian Apply
        if any(s >= 1.5 for s in geom_ord):
            HessApply = None

        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s > 1 for s in geom_ord):
            # eigs = (np.array([1., 0.1]), np.array([np.ones_like(x),-np.ones_like(x)]))
            # eigs = (np.ones(1), np.ones_like(x))
            eigs = self.eigdecomp(x, k=np.min([self.dimension, k]))

        return loglik, agrad, HessApply, eigs

    # define plot the solution
    def plotsolution(self, t, solution, solution_opt=None, filename_prex=None):
        # plot solutions

        solution = self.grouping(solution)

        plt.figure(1)
        plt.semilogy(t, solution[:, 0], 'b', label='$S$: susceptible')
        plt.semilogy(t, solution[:, 1], 'g', label='$E$: exposed')
        plt.semilogy(t, solution[:, 2], 'r', label='$Q$: quarantined')
        plt.semilogy(t, solution[:, 3], 'k', label='$A$: unconfirmed')
        plt.semilogy(t, solution[:, 4], 'm', label='$I$: confirmed')
        plt.semilogy(t, solution[:, 5], 'y', label='$H$: hospitalized')
        plt.semilogy(t, solution[:, 6], 'g.-', label='$R$: recovered')
        plt.semilogy(t, solution[:, 7], 'c', label='$D$: deceased')
        plt.semilogy(t, solution[:, 8], 'm.-', label='$T_c$: total confirmed')
        plt.semilogy(t, solution[:, 9], 'k.-', label='$T_u$: total unconfirmed')
        plt.legend(loc='best')
        plt.xlabel('time t (days)')
        plt.ylabel('# cases')
        plt.grid()
        if filename_prex is not None:
            filename = filename_prex + "all_compartments.pdf"
            plt.savefig(filename)

        plt.figure(2)
        plt.semilogy(t, solution[:, 5], 'y', label='$H$: hospitalized')
        plt.semilogy(t, solution[:, 7], 'c', label='$D$: deceased')
        # plt.semilogy(t, solution[:, 8], 'm.-', label='$T_c$: total confirmed')
        # plt.semilogy(t, solution[:, 9], 'k.-', label='$T_u$: total unconfirmed')
        plt.legend(loc='best')
        plt.xlabel('time t (days)')
        plt.ylabel('# cases')
        plt.grid()
        if filename_prex is not None:
            filename = filename_prex + "hospitalized_deceased.pdf"
            plt.savefig(filename)

        if solution_opt is not None:

            solution_opt = self.grouping(solution_opt)

            plt.figure(1)
            plt.semilogy(t, solution_opt[:, 0], 'b--', label='$S$: susceptible')
            plt.semilogy(t, solution_opt[:, 1], 'g--', label='$E$: exposed')
            plt.semilogy(t, solution_opt[:, 2], 'r--', label='$Q$: quarantined')
            plt.semilogy(t, solution_opt[:, 3], 'k--', label='$A$: unconfirmed')
            plt.semilogy(t, solution_opt[:, 4], 'm--', label='$I$: confirmed')
            plt.semilogy(t, solution_opt[:, 5], 'y--', label='$H$: hospitalized')
            plt.semilogy(t, solution_opt[:, 6], 'g.--', label='$R$: recovered')
            plt.semilogy(t, solution_opt[:, 7], 'c--', label='$D$: deceased')
            plt.semilogy(t, solution_opt[:, 8], 'm:', label='$T_c$: total confirmed')
            plt.semilogy(t, solution_opt[:, 9], 'k:', label='$T_u$: total unconfirmed')

            plt.legend(loc='best')
            plt.xlabel('time t (days)')
            plt.ylabel('# cases')
            plt.grid()
            if filename_prex is not None:
                filename = filename_prex + "all_compartments.pdf"
                plt.savefig(filename)

            plt.figure(2)
            plt.semilogy(t, solution_opt[:, 5], 'y--', label='$H$: hospitalized')
            plt.semilogy(t, solution_opt[:, 7], 'c--', label='$D$: deceased')
            # plt.semilogy(t, solution_opt[:, 8], 'm:', label='$T_c$: total confirmed')
            # plt.semilogy(t, solution_opt[:, 9], 'k:', label='$T_u$: total unconfirmed')
            plt.legend(loc='best')
            plt.xlabel('time t (days)')
            plt.ylabel('# cases')
            plt.grid()
            if filename_prex is not None:
                filename = filename_prex + "hospitalized_deceased.pdf"
                plt.savefig(filename)

        # plt.show()
        # plt.close()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fips", nargs='?', const=36, default=36, type=int, help="specify fips of states")
args = parser.parse_args()

# Texas: 48, California: 6, New York: 36, New Jersey: 34
fips = args.fips

states = pickle.load(open("utils/states_dictionary_moving_average", 'rb'))

# population 0-4, 5-9, ..., 80-84, 85+
population = states[fips]['population']
N_total = np.sum(population)

# print("fips = ", fips, "population = ", N_total)

state_name = states[fips]['name']

model_type = "scalar"

if model_type is "scalar":

    filename_prex = "figure/initial_scalar_" + state_name + "_"
    savefilename = "data/initial_scalar_solution_" + state_name

opt_data = np.load(savefilename+"_full.npz", allow_pickle=True)

configurations = opt_data["configurations"]
controls = opt_data["controls_opt"]
simulation_first_confirmed = opt_data["simulation_first_confirmed"]
parameters = opt_data["parameters_opt"]
# y0, t_total, N_total, number_group, population_proportion, \
# t_control, number_days_per_control_change, number_control_change_times, number_time_dependent_controls = configurations

alpha = np.arctanh(2*controls[0]-1)
controls = (alpha, )

misfit = Misfit(configurations, parameters, controls, simulation_first_confirmed)

# sigma = 10 * np.ones_like(x)
prior = Laplacian(misfit.dimension, gamma=10, mean=alpha)
# prior = Laplacian(misfit.dimension, gamma=10, regularization=False)

model = Model(prior, misfit)

if __name__ ==  "__main__":

    print(misfit.t_total)

    solution = RK2(misfit.seir, misfit.y0, misfit.t_total, misfit.parameters, misfit.controls)
    solution = misfit.grouping(solution)

    # compute growth rate at properly chosen time intervals
    t0, t1 = 10, 20
    E0, E1 = solution[t0, 1], solution[t1, 1]
    print("exposed growth rate beta for exp(beta*t) = ", np.log(E1/E0)/(t1-t0), "doubling time = ", np.log(2)/(np.log(E1/E0)/(t1-t0)))
    t0, t1 = 20, 30
    I0, I1 = solution[t0, 4], solution[t1, 4]
    print("symptomatic growth rate beta for exp(beta*t) = ", np.log(I1/I0)/(t1-t0), "doubling time = ", np.log(2)/(np.log(I1/I0)/(t1-t0)))
    t0, t1 = 30, 40
    D0, D1 = solution[t0, 7], solution[t1, 7]
    print("deceased growth rate beta for exp(beta*t) = ", np.log(D1/D0)/(t1-t0), "doubling time = ", np.log(2)/(np.log(D1/D0)/(t1-t0)))

    misfit.plotsolution(misfit.t_total, solution)

    # plt.show()

    from scipy.optimize import minimize, NonlinearConstraint, BFGS, Bounds

    # lower_bound = [0.0]
    # upper_bound = [0.95]
    lower_bound = [-3.]
    upper_bound = [3.]

    for i_p in range(1, len(misfit.controls)):
        parameter = misfit.controls[i_p]
        # parameter_l = parameter-2*parameter/4
        # parameter_u = parameter+2*parameter/4
        parameter_l = parameter / 2
        parameter_u = parameter * 2
        lower_bound.append(parameter_l)
        upper_bound.append(parameter_u)
    # lower_bound = [lower_bound[i] * ones for i in range(len(lower_bound))]
    # upper_bound = [upper_bound[i] * ones for i in range(len(upper_bound))]

    # lower_bound = [0.0 * ones, 0.0 * ones, 0.065 * ones, 0.001 * ones, 0.1 * ones]
    # upper_bound = [0.95 * ones, 0.2 * ones, 0.8 * ones, 0.02 * ones, 0.7 * ones]

    # lower_bound = [0.0 * ones, 0.0 * ones, 0.065 * ones, 0.0065 * ones, 0.2 * ones]
    # upper_bound = [0.95 * ones, 0.2 * ones, 0.8 * ones, 0.0065 * ones, 0.2 * ones]

    # lower_bound = [0.0]
    # upper_bound = [0.95]

    lower_bounds = np.array([np.tile(lower_bound[i], misfit.dimension) for i in range(misfit.number_time_dependent_controls)]).flatten()
    # for i in range(misfit.number_time_dependent_controls, len(lower_bound)):
    #     lower_bounds = np.append(lower_bounds, lower_bound[i])

    upper_bounds = np.array([np.tile(upper_bound[i], misfit.dimension) for i in range(misfit.number_time_dependent_controls)]).flatten()
    # for i in range(misfit.number_time_dependent_controls, len(upper_bound)):
    #     upper_bounds = np.append(upper_bounds, upper_bound[i])

    bounds = Bounds(lower_bounds, upper_bounds, keep_feasible=True)

    x0, _ = flatten(misfit.controls)

    print("x0 = ", x0.shape, "lower_bounds = ", lower_bounds.shape)

    OptRes = minimize(fun=misfit.cost, x0=x0, method="SLSQP", jac=misfit.gx,
                      bounds=bounds,
                      options={'maxiter': 50, 'iprint': 10, 'disp': True})

    print("OptRes = ", OptRes)

    # compute the optimal solution
    controls_opt = misfit.unflatten(OptRes["x"])
    solution_opt = RK2(misfit.seir, misfit.y0, misfit.t_total, misfit.parameters, controls_opt)
    parameters_opt = misfit.parameters
    # save data
    today = misfit.simulation_first_confirmed + len(misfit.data_confirmed)
    np.savez(savefilename, configurations=configurations, parameters=misfit.parameters,
             simulation_first_confirmed=misfit.simulation_first_confirmed, today=today,
             solution=solution, controls=misfit.controls,
             solution_opt=solution_opt, controls_opt=controls_opt, parameters_opt=parameters_opt)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # alpha = OptRes["x"]
    # # alpha = (np.tanh(alpha)+1)/2
    # plt.plot(alpha, '.-')
    # plt.show()

    # dimension = 101
    # prior = BrownianMotion(dimension)
    # noise = np.random.normal(0, 1, dimension)
    # x = np.zeros(dimension)
    # prior.sample(noise, x)
    #
    # beta = 10.
    # misfit = Misfit(dimension, beta)
    # u = misfit.solution(x)
    # loc = np.arange(0, dimension, 5)
    # sigma = 0.1
    # obs = u[loc] + np.random.normal(0, sigma, len(loc))
    # noise_covariance = np.diag(sigma**2*np.ones(len(loc)))
    # misfit = Misfit(dimension, beta, obs, loc, noise_covariance)
    #
    # misfit.x = x
    # model = Model(prior, misfit)
    #
    # d, U = misfit.eigdecomp(x, k=dimension-1)
    #
    # print("d, U = ", d, U)
    #
    # cost = misfit.cost(x)
    # g = np.zeros(dimension)
    # gx = misfit.grad(x, g)
    # xhat = np.ones(dimension)
    # h = np.zeros(dimension)
    # misfit.hvp.update_x(x)
    # misfit.hvp.mult(xhat, h)
    # print("cost, grad = ", cost, g, gx, "hvp", h)
    #
    # mg = np.zeros(dimension)
    # model.gradient(x, mg, misfit_only=False)
    # print("model cost, grad = ", model.cost(x), mg)
    #
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # nproc = comm.Get_size()
    # print("rank, nproc, comm", rank, nproc, comm)
    #
    # options["number_particles"] = 10
    # particle = Particle(model, options, comm)
    # print(particle.particles)
    #
    # particle.statistics()
    #
    # particle.communication()

    # n = 101
    # x0 = np.linspace(-2, 2, n)
    # x1 = np.linspace(-2, 2, n)
    # cost = np.zeros((n, n))
    #
    # for i in range(n):
    #     for j in range(n):
    #         x = [x0[i], x1[j]]
    #         cost[i, j] = misfit.post(x)
    #
    # x1, x0 = np.meshgrid(x0, x1)
    # plt.figure()
    # plt.contour(x0, x1, cost)
    # plt.show()
    #
