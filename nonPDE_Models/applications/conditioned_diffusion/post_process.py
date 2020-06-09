import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open("data/backup6/data_nSamples_128_isProjection_True_SVGD.p", 'rb'))
d_average = data["d_average"]
plt.figure()
for i in range(10):
    label = "$\ell = $"+str(i*10)
    plt.plot(np.log10(np.sort(d_average[i*10])[::-1]), '.-', label=label)
plt.xlabel("r", fontsize=16)
plt.ylabel(r"$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.savefig("figure/eigenvalues.pdf")
plt.show()


data = pickle.load(open("data/backup6/observation.p", "rb"))

dimension, beta, x, u, obs, loc, noise_covariance = data

from model import Misfit
misfit = Misfit(dimension, beta, obs, loc, noise_covariance)

t = np.linspace(0, 1, dimension+1)
x = np.append(0, x)

# plt.figure()
# plt.plot(t, u, '.-')
# plt.show()
MCMC = False
if MCMC:
    data = pickle.load(open("result/DILI_LI_Langevin_dim101_2020-05-31-23-07-34.pckl",'rb'))
    samples = data[2]
    # print(samples[0, :])
    plt.figure()
    for i in range(100):
        plt.plot(t, samples[-i*100,:],'--')
    plt.plot(t, x, 'r.-', linewidth=5)
    plt.show()

    # # plot total
    fix, ax = plt.subplots()

    total = []
    number_sample = 100
    for i in range(number_sample):
        sample = samples[-i*10,:]
        # ax.plot(t, sample, '--', linewidth=0.5)
        total.append(sample)
    total = np.array(total)
    dim = len(total[0, :])
    total_average = np.zeros(dim)
    total_plus = np.zeros(dim)
    total_minus = np.zeros(dim)
    for i in range(dim):
        total_sort = np.sort(total[:, i])
        # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
        total_average[i], total_plus[i], total_minus[i] = \
            np.mean(total_sort), total_sort[np.int(2.5 / 100 * number_sample)], total_sort[
                np.int(97.5 / 100 * number_sample)]

    ax.plot(t, total_average, '.-', linewidth=4)
    ax.plot(t, x, 'r.-', linewidth=4)
    ax.fill_between(t, total_minus, total_plus, color='gray', alpha=.2)

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # ax.legend(loc='best')
    ax.grid(True)

    plt.show()
    # plt.legend()
    # plt.title(total_titles[ind])
    # filename = filename_prex + total_titles[ind] + ".pdf"
    # plt.savefig(filename)

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
        total_average[id], total_plus[id], total_minus[id] = \
            np.mean(total_sort), total_sort[np.int(5 / 100 * number_sample)], total_sort[
                np.int(95 / 100 * number_sample)]

    ax.plot(t, total_average, '.-', linewidth=2, label="mean")
    ax.fill_between(t, total_minus, total_plus, color='gray', alpha=.2)


for i in range(20):
    ax1 = plt.subplot(2,2,1)
    # ax1.plot(t, x, 'r.-', linewidth=5)
    plt.title("SVGD sample")
    ax2 = plt.subplot(2,2,2)
    # ax2.plot(t, u, 'r.-', linewidth=5)
    # ax2.plot(t[loc], obs, 'o', markersize=12)
    plt.title("SVGD solution")

    ax3 = plt.subplot(2,2,3)
    # ax3.plot(t, x, 'r.-', linewidth=5)
    plt.xlabel("pSVGD sample")
    ax4 = plt.subplot(2,2,4)
    # ax4.plot(t, u, 'r.-', linewidth=5)
    # ax4.plot(t[loc], obs, 'o', markersize=12)
    plt.xlabel("pSVGD solution")

    filename = "data/backup6/particle_isProjection_False_"+"iteration_"+str(i*10)+".npz"
    data = np.load(filename)
    particles_SVGD = data["particle"]

    filename = "data/backup6/particle_isProjection_True_"+"iteration_"+str(i*10)+".npz"
    data = np.load(filename)
    particles_pSVGD = data["particle"]

    shape = particles_SVGD.shape
    print("i = ", i, "particles shape = ", shape)
    samples_pSVGD = []
    solutions_pSVGD = []
    samples_SVGD = []
    solutions_SVGD = []
    for j in range(shape[0]):
        for k in range(shape[1]):
            # plt.subplot(2,2,1)
            sample = particles_SVGD[j, k, :]
            # ax1.plot(t, sample, '--')
            # plt.subplot(2,2,2)
            solution = misfit.solution(sample)
            # ax2.plot(t, solution, '--')
            sample = np.append(0, particles_SVGD[j, k, :])
            samples_SVGD.append(sample)
            solutions_SVGD.append(solution)

            # plt.subplot(2,2,3)
            sample = particles_pSVGD[j, k, :]
            # ax3.plot(t, sample, '--')
            # plt.subplot(2,2,4)
            solution = misfit.solution(sample)
            # ax4.plot(t, solution, '--')
            sample = np.append(0, particles_pSVGD[j, k, :])
            samples_pSVGD.append(sample)
            solutions_pSVGD.append(solution)

    # plt.figure(1)
    # plt.title("samples")
    # plt.figure(2)
    # plt.title("solutions")

    # total = samples_SVGD
    # dim = len(total[0, :])
    # total_average = np.zeros(dim)
    # total_plus = np.zeros(dim)
    # total_minus = np.zeros(dim)
    # for id in range(dim):
    #     total_sort = np.sort(total[:, id])
    #     # total_average[i], total_plus[i], total_minus[i] = mean_confidence_interval(total[:,i])
    #     total_average[id], total_plus[id], total_minus[id] = \
    #         np.mean(total_sort), total_sort[np.int(5 / 100 * number_sample)], total_sort[
    #             np.int(95 / 100 * number_sample)]
    #
    # ax1.plot(t, total_average, '.-', linewidth=4, label="sample mean")

    plot_total(t, np.array(samples_SVGD), ax1)
    ax1.plot(t, x, 'r.-', linewidth=2, label="true")
    # ax1.plot(t, x, 'r.-', linewidth=3)
    # plt.title("SVGD samples")
    # filename = "figure/SVGD_sample_"+str(i*10)+".pdf"
    # plt.savefig(filename)
    ax1.legend()

    plot_total(t, np.array(solutions_SVGD), ax2)
    ax2.plot(t, u, 'r.-', linewidth=2, label="true")
    ax2.plot(t[loc], obs, 'ko', markersize=4, label="noisy data")
    # plt.title("solution at SVGD samples")
    # filename = "figure/SVGD_solution_"+str(i*10)+".pdf"
    # plt.savefig(filename)
    ax2.legend()

    plot_total(t, np.array(samples_pSVGD), ax3)
    ax3.plot(t, x, 'r.-', linewidth=2, label="true")
    # ax1.plot(t, x, 'r.-', linewidth=3)
    # plt.title("SVGD samples")
    # filename = "figure/pSVGD_sample_" + str(i*10) + ".pdf"
    # plt.savefig(filename)
    ax3.legend()

    plot_total(t, np.array(solutions_pSVGD), ax4)
    ax4.plot(t, u, 'r.-', linewidth=2, label="true")
    ax4.plot(t[loc], obs, 'ko', markersize=4, label="noisy data")
    # plt.title("solution at SVGD samples")
    # filename = "figure/pSVGD_solution_" + str(i*10) + ".pdf"
    # plt.savefig(filename)
    ax4.legend()

    filename = "figure/pSVGDvsSVGD_" + str(i*10) + ".pdf"

    plt.savefig(filename)

    # ax3.plot(t, x, 'r.-', linewidth=3)
    # # plt.title("pSVGD samples")
    # ax4.plot(t, u, 'r.-', linewidth=3)
    # ax4.plot(t[loc], obs, 'ko', markersize=6)
    # # plt.title("solution at pSVGD samples")

    # plt.show()
    plt.close("all")
