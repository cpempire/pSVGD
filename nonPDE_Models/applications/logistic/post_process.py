import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

data = pickle.load(open("data/observation.p", "rb"))

# dimension, obs, noise_covariance = data

from model import *

misfit.inputs = inputs_testing

data = pickle.load(open("data/data_nSamples_32_isProjection_True_SVGD.p", 'rb'))
d_average = data["d_average"]
plt.figure()
step = 100
for i in range(np.floor_divide(len(d_average), step)):
    label = "$\ell = $"+str(i*step)
    plt.plot(np.log10(np.sort(d_average[i*step])[::-1]), '.-', label=label)
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.savefig("figure/logistic_eigenvalues.pdf")
# plt.show()
plt.close()

# misfit = Misfit(dimension, beta, obs, loc, noise_covariance)
#
# t = np.linspace(0, 1, dimension)
#
# # plt.figure()
# # plt.plot(t, u, '.-')
# # plt.show()
#
# data = pickle.load(open("result/DILI_LI_Langevin_dim101_2020-05-31-23-07-34.pckl",'rb'))
# samples = data[2]
# # print(samples[0, :])
# plt.figure()
# for i in range(100):
#     plt.plot(t, samples[-i*100,:],'--')
# plt.plot(t, x, 'r.-', linewidth=5)
# plt.show()
#
# # # plot total
# fix, ax = plt.subplots()
#
# total = []
# number_sample = 100
# for i in range(number_sample):
#     sample = samples[-i*10,:]
#     # ax.plot(t, sample, '--', linewidth=0.5)
#     total.append(sample)
# total = np.array(total)
# dim = len(total[0, :])
# total_average = np.zeros(dim)
# total_plus = np.zeros(dim)
# total_minus = np.zeros(dim)
# for i in range(dim):
#     total_sort = np.sort(total[:, i])
#     # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
#     total_average[i], total_plus[i], total_minus[i] = \
#         np.mean(total_sort), total_sort[np.int(2.5 / 100 * number_sample)], total_sort[
#             np.int(97.5 / 100 * number_sample)]
#
# ax.plot(t, total_average, '.-', linewidth=4)
# ax.plot(t, x, 'r.-', linewidth=4)
# ax.fill_between(t, total_minus, total_plus, color='gray', alpha=.2)
#
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# # ax.legend(loc='best')
# ax.grid(True)
#
# plt.show()
# # plt.legend()
# # plt.title(total_titles[ind])
# # filename = filename_prex + total_titles[ind] + ".pdf"
# # plt.savefig(filename)


def plot_total(samples, ax):

    total = samples
    number_sample = len(total)
    dim = len(total[0, :])
    total_average = np.zeros(dim)
    total_plus = np.zeros(dim)
    total_minus = np.zeros(dim)
    for id in range(dim):
        total_sort = np.sort(total[:, id])
        # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
        total_average[id], total_plus[id], total_minus[id] = \
            np.mean(total_sort), total_sort[np.int(5 / 100 * number_sample)], total_sort[
                np.int(95 / 100 * number_sample)]

    ax.plot(total_average, '.', linewidth=2, label="mean")
    # ax.fill_between(np.arange(len(total_average)), total_minus, total_plus, color='gray', alpha=.2)

    for i in range(dim):
        ax.fill_between([i,i], [total_minus[i], total_minus[i]], [total_plus[i],total_plus[i]], color='gray', alpha=.2)


for i in range(11):
    filename = "data/particle_isProjection_False_"+"iteration_"+str(i*step)+".npz"
    data = np.load(filename)
    particles_SVGD = data["particle"]

    filename = "data/particle_isProjection_True_"+"iteration_"+str(i*step)+".npz"
    data = np.load(filename)
    particles_pSVGD = data["particle"]

    shape = particles_SVGD.shape
    print("i = ", i, "particles shape = ", shape)

    solutions_pSVGD = []
    cost_pSVGD = []
    solutions_SVGD = []
    cost_SVGD = []
    for j in range(shape[0]):
        for k in range(shape[1]):
            sample = particles_SVGD[j, k, :]
            solution = misfit.forward(sample)
            cost = misfit.cost(sample)
            solutions_SVGD.append(solution)
            cost_SVGD.append(cost)

            sample = particles_pSVGD[j, k, :]
            solution = misfit.forward(sample)
            cost = misfit.cost(sample)
            solutions_pSVGD.append(solution)
            cost_pSVGD.append(cost)

    solutions_SVGD = np.array(solutions_SVGD)
    solutions_pSVGD = np.array(solutions_pSVGD)
    cost_SVGD = np.array(cost_SVGD)
    cost_pSVGD = np.array(cost_pSVGD)

    # ax1 = plt.subplot(2,2,1)
    # plt.title("SVGD log-likelihood")
    # plt.plot(cost_SVGD, ax1)

    ax1 = plt.subplot(2,1,1)
    plot_total(solutions_SVGD, ax1)
    ax1.plot(outputs_testing, 'r.', linewidth=2, label="test data")
    ax1.legend(loc='upper right')

    outputs_forward = (np.mean(solutions_SVGD, axis=0) >= 0.5)
    accuracy = np.sum(np.abs(outputs_testing - outputs_forward) < 1e-3)/len(outputs_testing)
    title = "SVGD test accuracy = "+str(accuracy)
    plt.title(title, fontsize=16)

    ax2 = plt.subplot(2,1,2)
    plot_total(solutions_pSVGD, ax2)
    ax2.plot(outputs_testing, 'r.', linewidth=2, label="test data")
    ax2.legend(loc='upper right')

    outputs_forward = (np.mean(solutions_pSVGD, axis=0) >= 0.5)
    accuracy = np.sum(np.abs(outputs_testing - outputs_forward) < 1e-3)/len(outputs_testing)
    title = "pSVGD test accuracy = "+str(accuracy)
    plt.xlabel(title, fontsize=16)

    filename = "figure/logistic_pSVGDvsSVGD_" + str(i*step) + ".pdf"

    plt.savefig(filename)

    # ax3.plot(t, x, 'r.', linewidth=3)
    # # plt.title("pSVGD samples")
    # ax2.plot(t, u, 'r.', linewidth=3)
    # ax2.plot(t[loc], obs, 'ko', markersize=6)
    # # plt.title("solution at pSVGD samples")

    # plt.show()
    plt.close("all")




