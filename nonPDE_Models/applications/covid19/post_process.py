import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
# # plot data, moving average, optimized simulation, and prediction

data = pickle.load(open("data/data_nSamples_128_isProjection_True_SVGD.p", 'rb'))
d_average = data["d_average"]
plt.figure()
step = 20
for i in range(np.floor_divide(len(d_average), step)):
    label = "$\ell = $"+str(i*step)
    plt.plot(np.log10(np.sort(d_average[i*step])[::-1]), '.-', label=label)
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.savefig("figure/covid19_eigenvalues.pdf")
# plt.show()
plt.close()

from model import *

time_delta = timedelta(days=1)
stop_date = datetime(2020, 6, 5)
start_date = stop_date - timedelta(len(misfit.t_total))
dates = mdates.drange(start_date, stop_date, time_delta)


def plot_total(t, samples, ax):

    total = samples
    number_sample = len(total)
    dim = len(total[0, :])
    total_average = np.zeros(dim)
    total_plus = np.zeros(dim)
    total_minus = np.zeros(dim)
    for id in range(dim):
        total_sort = np.sort(total[:, id])
        # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
        id_1 = np.int(5 / 100 * number_sample)
        id_2 = np.int(95 / 100 * number_sample)
        total_average[id], total_plus[id], total_minus[id] = np.mean(total_sort[id_1:id_2]), total_sort[id_1], total_sort[id_2]

    ax.plot(t, total_average, '.-', linewidth=2, label="mean")
    ax.fill_between(t, total_minus, total_plus, color='gray', alpha=.2)


for i in range(20,21):

    filename = "data/particle_isProjection_False_"+"iteration_"+str(i*10)+".npz"
    data = np.load(filename)
    particles_SVGD = data["particle"]

    filename = "data/particle_isProjection_True_"+"iteration_"+str(i*10)+".npz"
    data = np.load(filename)
    particles_pSVGD = data["particle"]

    shape = particles_pSVGD.shape
    print("i = ", i, "particles shape = ", shape)
    samples_pSVGD = []
    solutions_pSVGD = []
    samples_SVGD = []
    solutions_SVGD = []
    for j in range(shape[0]):
        for k in range(shape[1]):
            sample = particles_SVGD[j, k, :]
            solution = misfit.solution(sample)
            samples_SVGD.append((np.tanh(sample)+1)/2)
            solutions_SVGD.append(solution)

            sample = particles_pSVGD[j, k, :]
            solution = misfit.solution(sample)
            samples_pSVGD.append((np.tanh(sample)+1)/2)
            solutions_pSVGD.append(solution)

    solutions_SVGD = np.array(solutions_SVGD)
    solutions_pSVGD = np.array(solutions_pSVGD)
    samples_SVGD = np.array(samples_SVGD)
    samples_pSVGD = np.array(samples_pSVGD)

    # # # plot samples and solutions at one figure
    # ax1 = plt.subplot(2,2,1)
    # plt.title("SVGD sample")
    # plot_total(dates, samples_SVGD, ax1)
    # ax1.legend()
    #
    # ax2 = plt.subplot(2,2,2)
    # plt.title("SVGD solution")
    # plot_total(dates, solutions_SVGD, ax2)
    # ax2.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=4, label="data")
    # ax2.legend()
    #
    # ax3 = plt.subplot(2,2,3)
    # plt.xlabel("pSVGD sample")
    # plot_total(dates, samples_pSVGD, ax3)
    # ax3.legend()
    #
    # ax4 = plt.subplot(2,2,4)
    # plt.xlabel("pSVGD solution")
    # plot_total(dates, solutions_pSVGD, ax4)
    # ax4.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=4, label="data")
    # ax4.legend()
    #
    # filename = "figure/pSVGDvsSVGD_" + str(i*10) + ".pdf"
    # plt.savefig(filename)
    #
    # plt.close("all")

    # # plot samples
    ax1 = plt.subplot(2, 1, 1)
    plot_total(dates, samples_SVGD, ax1)
    ax1.plot(dates, opt_data["controls_opt"][0], 'k.-', label="optimal")
    plt.title("SVGD social distancing", fontsize=16)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.grid(True)

    ax2 = plt.subplot(2, 1, 2)
    plot_total(dates, samples_pSVGD, ax2)
    ax2.plot(dates, opt_data["controls_opt"][0], 'k.-', label="optimal")
    plt.xlabel("pSVGD social distancing", fontsize=16)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.grid(True)

    filename = "figure/samples_pSVGDvsSVGD_" + str(i*10) + ".pdf"
    plt.savefig(filename)

    plt.close("all")

    # # plot solutions
    ax1 = plt.subplot(2, 1, 1)
    plot_total(dates, solutions_SVGD, ax1)
    ax1.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=2, label="data")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.grid(True)
    plt.title("SVGD # hospitalized", fontsize=16)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    plot_total(dates, solutions_pSVGD, ax2)
    ax2.plot(dates[misfit.loc], misfit.data_hospitalized, 'ko', markersize=2, label="data")

    plt.xlabel("pSVGD # hospitalized", fontsize=16)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.grid(True)

    filename = "figure/solutions_pSVGDvsSVGD_" + str(i * 10) + ".pdf"
    plt.savefig(filename)
