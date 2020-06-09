import pickle
import numpy as np
import matplotlib.pyplot as plt

dSet = np.array([8, 16, 32, 64])
dPara = [81, 289, 1089, 4225]
NSet = [256, 256, 256, 256]

makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']
makerRMSEFalse = ['bs--', 'r*--', 'y<--', 'b>--']

labelTrue = ['pSVGD d=81', 'pSVGD d=289', 'pSVGD d=1,089', 'pSVGD d=4,225']
labelFalse = ['SVGD d=81', 'SVGD d=289', 'SVGD d=1,089', 'SVGD d=4,225']
fig = plt.figure(1)
for case in [0, 1, 2, 3]:

    d, N = dSet[case], NSet[case]

    filename = "accuracy-" + str(d) + 'x' + str(d) + "/data/data_nSamples_" + str(N) + "_isProjection_" + str(True) + "_SVGD_5" + ".p"

    data = pickle.load(open(filename, 'rb'))

    # print("d", d)

    plt.plot(np.log10(data["step_norm"]/data["step_norm"][0]), makerRMSETrue[case], label=labelTrue[case])

    # filename = "accuracy-" + str(d) + 'x' + str(d) + "/data/data_nSamples_" + str(N) + "_isProjection_" + str(False) + "_SVGD_5" + ".p"
    #
    # data = pickle.load(open(filename, 'rb'))
    #
    # # print("d", d)
    #
    # plt.plot(np.log10(data["step_norm"]/data["step_norm"][0]), makerRMSEFalse[case], label=labelFalse[case])

plt.xlabel("# iterations", fontsize=16)
plt.ylabel("$\log_{10}$(averaged step norm)", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/convergence_dimension.pdf"
fig.savefig(filename, format='pdf')