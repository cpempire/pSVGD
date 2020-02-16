import pickle
import numpy as np
import matplotlib.pyplot as plt


# d = 65
#
# filename = "data/laplace_nDimension_"+str(d)+".p"
# data_laplace = pickle.load(open(filename,'rb'))
# sample_statistics = data_laplace["sample_statistics"]
# qoi_statistics = data_laplace["qoi_statistics"]
# true_statistics = data_laplace["true_statistics"]
#
# true_meanL2norm, true_varianceL2norm, true_trace = true_statistics[0], true_statistics[1], true_statistics[2]
# # print("true_meanL2norm = ", true_meanL2norm, "true_varianceL2norm = ", true_varianceL2norm, "true_trace = ", true_trace)
#
# nSamples = np.zeros(5)
# laplace_meanL2norm = np.zeros(5)
# laplace_varianceL2norm = np.zeros(5)
# laplace_trace = np.zeros(5)
# laplace_qoi_mean = np.zeros(5)
# laplace_qoi_std = np.zeros(5)
#
#
# for iter in range(5):
#     # print("sample_statistics[iter][0]", sample_statistics[iter][0])
#     nSamples[iter] = sample_statistics[iter][0]
#     laplace_meanL2norm[iter] = sample_statistics[iter][1]
#     laplace_varianceL2norm[iter] = sample_statistics[iter][2]
#     laplace_trace[iter] = sample_statistics[iter][3]
#     laplace_qoi_mean[iter] = qoi_statistics[iter][1]
#     laplace_qoi_std[iter] = qoi_statistics[iter][2]

    # print("nSamples = ", nSamples[iter], "laplace_meanL2norm = ", laplace_meanL2norm[iter], "laplace_varianceL2norm = ",
    #       laplace_varianceL2norm[iter], "laplace_trace = ", laplace_trace[iter],
    #       "laplace_qoi_mean = ", laplace_qoi_mean[iter], "laplace_qoi_std = ", laplace_qoi_std[iter])


dSet = [16, 64, 256, 1024]
dPara = np.array(dSet)
NSet = [256, 256, 256, 256]
# NSet = [64, 64, 64, 64]
# NSet = [16, 16, 16, 16]

Ntrial = 10
Niter = 200

makerTrue = ['g.--', 'k.--', 'm.--', 'c.--']
makerFalse = ['b.--', 'r.--', 'y.--', 'b.--']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']
makerRMSEFalse = ['bs-', 'r*-', 'y<-', 'b>-']

#################################### SVN

meanErrorL2normTrue = np.zeros((4, Ntrial, Niter))
varianceErrorL2normTrue = np.zeros((4, Ntrial, Niter))
meanErrorL2normFalse = np.zeros((4, Ntrial, Niter))
varianceErrorL2normFalse = np.zeros((4, Ntrial, Niter))
meanErrorL2normSVGD = np.zeros((4, Ntrial, Niter))
varianceErrorL2normSVGD = np.zeros((4, Ntrial, Niter))

labelTrue = ['pSVGD d=17', 'pSVGD d=65', 'pSVGD d=257', 'pSVGD d=1025']
labelFalse = ['SVGD d=17', 'SVGD d=65', 'SVGD d=257', 'SVGD d=1025']
fig1 = plt.figure(1)
fig2 = plt.figure(2)
for case in [0, 1, 2, 3]:

    d, N = dSet[case], NSet[case]

    for i in range(Ntrial):
        filename = "accuracy-" + str(d) + "/data/data_nSamples_" + str(N) + "_isProjection_" + str(True) + "_SVGD_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
        varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
        meanErrorL2normTrue[case, i, :] = meanErrorL2normTrue_i
        varianceErrorL2normTrue[case, i, :] = varianceErrorL2normTrue_i

        filename = "accuracy-" + str(d) + "/data/data_nSamples_" + str(N) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"

        data_save = pickle.load(open(filename, 'rb'))
        meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
        varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
        meanErrorL2normFalse[case, i, :] = meanErrorL2normFalse_i
        varianceErrorL2normFalse[case, i, :] = varianceErrorL2normFalse_i

        # filename = "accuracy-" + str(d) + "/data/data_nSamples_" + str(N) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"
        #
        # data_save = pickle.load(open(filename, 'rb'))
        # meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
        # varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
        # meanErrorL2normSVGD[case, i, :] = meanErrorL2normFalse_i
        # varianceErrorL2normSVGD[case, i, :] = varianceErrorL2normFalse_i

    plt.figure(1)
    for i in range(Ntrial):
        plt.plot(np.log10(meanErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(meanErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normFalse[case,:,:]**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
    plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normTrue[case,:,:]**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])

    plt.figure(2)
    for i in range(Ntrial):
        plt.plot(np.log10(varianceErrorL2normFalse[case, i, :]), makerFalse[case], alpha=0.2)
        plt.plot(np.log10(varianceErrorL2normTrue[case, i, :]), makerTrue[case], alpha=0.2)
    plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normFalse[case,:,:]**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
    plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normTrue[case,:,:]**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])

plt.figure(1)
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_mean_"+"SVGD.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_"+"SVGD.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()

plt.figure(2)
plt.legend(fontsize=16)
plt.xlabel("# iterations", fontsize=16)
plt.ylabel("Log10(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/error_variance_" + "SVGD.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_" + "SVGD.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


meanErrorDimensionTrue = meanErrorL2normTrue[:, :, -1]
meanErrorDimensionFalse = meanErrorL2normFalse[:, :, -1]
varianceErrorDimensionTrue = varianceErrorL2normTrue[:, :, -1]
varianceErrorDimensionFalse = varianceErrorL2normFalse[:, :, -1]
# meanErrorDimensionSVGD = meanErrorL2normSVGD[:, :, 1]
# varianceErrorDimensionSVGD = varianceErrorL2normSVGD[:, :, 1]

case = 0

fig1 = plt.figure(1)
for i in range(Ntrial):
    # plt.plot(np.log2(dPara), np.log10(meanErrorDimensionSVGD[:, i]), makerFalse[1], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(meanErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)

# plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionSVGD ** 2, axis=1))), makerRMSEFalse[1], linewidth=2,
#          label='SVGD')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='SVGD')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(meanErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pSVGD')
plt.figure(1)
plt.legend(fontsize=16)
plt.xlabel("$\log_2(d-1)$", fontsize=16)
plt.ylabel("$\log_{10}$(RMSE of mean)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/error_mean_dimension"+"SVGD.pdf"
fig1.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_mean_dimension"+"SVGD.eps"
fig1.savefig(filename, format='eps', bbox_inches='tight')

plt.close()


fig2 = plt.figure(2)
for i in range(Ntrial):
    # plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionSVGD[:, i]), makerFalse[1], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionFalse[:, i]), makerFalse[case], alpha=0.2)
    plt.plot(np.log2(dPara), np.log10(varianceErrorDimensionTrue[:, i]), makerTrue[case], alpha=0.2)
# plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionSVGD ** 2, axis=1))), makerRMSEFalse[1], linewidth=2,
#          label='SVGD')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionFalse ** 2, axis=1))), makerRMSEFalse[case], linewidth=2,
         label='SVGD')
plt.plot(np.log2(dPara), np.log10(np.sqrt(np.mean(varianceErrorDimensionTrue ** 2, axis=1))), makerRMSETrue[case], linewidth=2,
         label='pSVGD')
plt.legend(fontsize=16)
plt.xlabel("$\log_2(d-1)$", fontsize=16)
plt.ylabel("$\log_{10}$(RMSE of variance)", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/error_variance_dimension"+"SVGD.pdf"
fig2.savefig(filename, format='pdf', bbox_inches='tight')
filename = "figure/error_variance_dimension"+"SVGD.eps"
fig2.savefig(filename, format='eps', bbox_inches='tight')

plt.close()
#################################### SVGD
#
# meanErrorL2normTrue = np.zeros((Ntrial, 100))
# varianceErrorL2normTrue = np.zeros((Ntrial, 100))
# meanErrorL2normFalse = np.zeros((Ntrial, 100))
# varianceErrorL2normFalse = np.zeros((Ntrial, 100))
#
# labelTrue = ['pSVGD 32', 'pSVGD 512']
# labelFalse = ['SVGD 32', 'SVGD 512']
# fig1 = plt.figure(1)
# fig2 = plt.figure(2)
# for case in [0, 1]:
#
#     p, N = pSet[case], NSet[case]
#
#     for i in range(Ntrial):
#         filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
#                    + str(d) + "_isProjection_" + str(True) + "_SVGD_" + str(i+1) + ".p"
#
#         data_save = pickle.load(open(filename, 'rb'))
#         meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
#         varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
#         meanErrorL2normTrue[i, :len(meanErrorL2normTrue_i)] = meanErrorL2normTrue_i
#         varianceErrorL2normTrue[i, :len(varianceErrorL2normTrue_i)] = varianceErrorL2normTrue_i
#
#         filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
#                    + str(d) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"
#
#         data_save = pickle.load(open(filename, 'rb'))
#         meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
#         varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
#         meanErrorL2normFalse[i, :len(meanErrorL2normFalse_i)] = meanErrorL2normFalse_i
#         varianceErrorL2normFalse[i, :len(varianceErrorL2normFalse_i)] = varianceErrorL2normFalse_i
#
#     plt.figure(1)
#     for i in range(Ntrial):
#         plt.plot(np.log10(meanErrorL2normFalse[i, :]), makerFalse[case], alpha=0.2)
#         plt.plot(np.log10(meanErrorL2normTrue[i, :]), makerTrue[case], alpha=0.2)
#     # plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normFalse**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
#     # plt.plot(np.log10(np.sqrt(np.mean(meanErrorL2normTrue**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])
#
#     plt.figure(2)
#     for i in range(Ntrial):
#         plt.plot(np.log10(varianceErrorL2normFalse[i, :]), makerFalse[case], alpha=0.2)
#         plt.plot(np.log10(varianceErrorL2normTrue[i, :]), makerTrue[case], alpha=0.2)
#     # plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normFalse**2, axis=0))), makerRMSEFalse[case], linewidth=2, label=labelFalse[case])
#     # plt.plot(np.log10(np.sqrt(np.mean(varianceErrorL2normTrue**2, axis=0))), makerRMSETrue[case], linewidth=2, label=labelTrue[case])
#
# plt.figure(1)
# plt.legend(fontsize=16)
# plt.xlabel("# iterations", fontsize=16)
# plt.ylabel("Log10(RMSE of mean)", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=16)
#
# filename = "figure/error_mean_"+"SVGD.pdf"
# fig1.savefig(filename, format='pdf', bbox_inches='tight')
# filename = "figure/error_mean_"+"SVGD.eps"
# fig1.savefig(filename, format='eps', bbox_inches='tight')
#
# plt.close()
#
# plt.figure(2)
# plt.legend(fontsize=16)
# plt.xlabel("# iterations", fontsize=16)
# plt.ylabel("Log10(RMSE of variance)", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=16)
#
# filename = "figure/error_variance_" + "SVGD.pdf"
# fig2.savefig(filename, format='pdf', bbox_inches='tight')
# filename = "figure/error_variance_" + "SVGD.eps"
# fig2.savefig(filename, format='eps', bbox_inches='tight')
#
# plt.close()



plot = False
if plot:

    meanL2normTrue = np.zeros((Ntrial, 4))
    varianceL2normTrue = np.zeros((Ntrial, 4))
    meanL2normFalse = np.zeros((Ntrial, 4))
    varianceL2normFalse = np.zeros((Ntrial, 4))

    for case in [1,3]:

        p, N = pSet[case], NSet[case]
        if case == 1:
            meanL2normFalse_set = np.zeros((Ntrial, 10))
            meanL2normTrue_set = np.zeros((Ntrial, 10))
            varianceL2normFalse_set = np.zeros((Ntrial, 10))
            varianceL2normTrue_set = np.zeros((Ntrial, 10))
            # meanL2normSVGDTrue_set = np.zeros((Ntrial, 100))
            # varianceL2normSVGDTrue_set = np.zeros((Ntrial, 100))
            # meanL2normSVGDFalse_set = np.zeros((Ntrial, 100))
            # varianceL2normSVGDFalse_set = np.zeros((Ntrial, 100))

        elif case == 3:
            meanL2normFalse_set3 = np.zeros((Ntrial, 10))
            meanL2normTrue_set3 = np.zeros((Ntrial, 10))
            varianceL2normFalse_set3 = np.zeros((Ntrial, 10))
            varianceL2normTrue_set3 = np.zeros((Ntrial, 10))
            # meanL2normSVGDTrue_set3 = np.zeros((Ntrial, 100))
            # varianceL2normSVGDTrue_set3 = np.zeros((Ntrial, 100))
            # meanL2normSVGDFalse_set3 = np.zeros((Ntrial, 100))
            # varianceL2normSVGDFalse_set3 = np.zeros((Ntrial, 100))

        for i in range(Ntrial):
            filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
                       + str(d) + "_isProjection_" + str(True) + "_SVN_" + str(i+1) + ".p"

            data_save = pickle.load(open(filename, 'rb'))
            meanL2normTrue_i = data_save["meanL2norm"]
            meanErrorL2normTrue_i = data_save["meanErrorL2norm"]
            # meanL2normTrue[i, case] = meanL2normTrue_i[-1]
            varianceL2normTrue_i = data_save["varianceL2norm"]
            varianceErrorL2normTrue_i = data_save["varianceErrorL2norm"]
            # varianceL2normTrue[i, case] = varianceL2normTrue_i[-1]

            filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
                       + str(d) + "_isProjection_" + str(False) + "_SVN_" + str(i+1) + ".p"

            data_save = pickle.load(open(filename, 'rb'))
            meanL2normFalse_i = data_save["meanL2norm"]
            meanErrorL2normFalse_i = data_save["meanErrorL2norm"]
            # meanL2normFalse[i, case] = meanL2normFalse_i[-1]
            varianceL2normFalse_i = data_save["varianceL2norm"]
            varianceErrorL2normFalse_i = data_save["varianceErrorL2norm"]
            # varianceL2normFalse[i, case] = varianceL2normFalse_i[-1]

            # filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
            #            + str(d) + "_isProjection_" + str(True) + "_SVGD_" + str(i+1) + ".p"
            #
            # data_save = pickle.load(open(filename, 'rb'))
            # meanL2normSVGDTrue_i = data_save["meanL2norm"]
            # meanL2normSVGDTrue[i, case] = meanL2normFalse_i[-1]
            # varianceL2normSVGDTrue_i = data_save["varianceL2norm"]
            # varianceL2normSVGDTrue[i, case] = varianceL2normFalse_i[-1]
            #
            # filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
            #            + str(d) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"
            #
            # data_save = pickle.load(open(filename, 'rb'))
            # meanL2normSVGDFalse_i = data_save["meanL2norm"]
            # meanL2normSVGDFalse[i, case] = meanL2normFalse_i[-1]
            # varianceL2normSVGDFalse_i = data_save["varianceL2norm"]
            # varianceL2normSVGDFalse[i, case] = varianceL2normFalse_i[-1]

            if case == 1:
                meanL2normTrue_set[i, :] = meanL2normTrue_i
                varianceL2normTrue_set[i, :] = varianceL2normTrue_i
                meanL2normFalse_set[i, :] = meanL2normFalse_i
                varianceL2normFalse_set[i, :] = varianceL2normFalse_i
                # meanL2normSVGDTrue_set[i, :] = meanL2normSVGDTrue_i
                # varianceL2normSVGDTrue_set[i, :] = varianceL2normSVGDTrue_i
                # meanL2normSVGDFalse_set[i, :] = meanL2normSVGDFalse_i
                # varianceL2normSVGDFalse_set[i, :] = varianceL2normSVGDFalse_i

            elif case == 3:
                meanL2normTrue_set3[i, :] = meanL2normTrue_i
                varianceL2normTrue_set3[i, :] = varianceL2normTrue_i
                meanL2normFalse_set3[i, :] = meanL2normFalse_i
                varianceL2normFalse_set3[i, :] = varianceL2normFalse_i
                # meanL2normSVGDTrue_set3[i, :] = meanL2normSVGDTrue_i
                # varianceL2normSVGDTrue_set3[i, :] = varianceL2normSVGDTrue_i
                # meanL2normSVGDFalse_set3[i, :] = meanL2normSVGDFalse_i
                # varianceL2normSVGDFalse_set3[i, :] = varianceL2normSVGDFalse_i

            fig = plt.figure()
            plt.plot(np.log10(np.abs(meanL2normFalse_i - true_meanL2norm)), 'bx-', label='SVN')
            plt.plot(np.log10(np.abs(meanL2normTrue_i - true_meanL2norm)), 'r.-', label='pSVN')
            plt.plot(np.log10(meanErrorL2normFalse_i), 'go-', label='SVN error')
            plt.plot(np.log10(meanErrorL2normTrue_i), 'kd-', label='pSVN error')

            plt.legend(fontsize=16)
            plt.xlabel("# iterations", fontsize=16)
            plt.ylabel("Log10(error of mean)", fontsize=16)

            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=16)

            filename = "figure/error_nSamples_"+str(NSet[case])+"_mean_"+str(i)+"SVN.pdf"
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            filename = "figure/error_nSamples_"+str(NSet[case])+"_mean_"+str(i)+"SVN.eps"
            fig.savefig(filename, format='eps', bbox_inches='tight')

            plt.close()

            fig = plt.figure()
            plt.plot(np.log10(np.abs(varianceL2normFalse_i - true_varianceL2norm)), 'bx-', label='SVN')
            plt.plot(np.log10(np.abs(varianceL2normTrue_i - true_varianceL2norm)), 'r.-', label='pSVN')
            plt.plot(np.log10(varianceErrorL2normFalse_i), 'go-', label='SVN error')
            plt.plot(np.log10(varianceErrorL2normTrue_i), 'kd-', label='pSVN error')

            plt.legend(fontsize=16)
            plt.xlabel("# iterations", fontsize=16)
            plt.ylabel("Log10(error of variance)", fontsize=16)

            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=16)

            filename = "figure/error_nSamples_" + str(NSet[case]) + "_variance_" + str(i) + "SVN.pdf"
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            filename = "figure/error_nSamples_" + str(NSet[case]) + "_variance_" + str(i) + "SVN.eps"
            fig.savefig(filename, format='eps', bbox_inches='tight')

            plt.close()

            # fig = plt.figure()
            # plt.plot(np.log10(np.abs(meanL2normSVGDFalse_i - true_meanL2norm)), 'bx-', label='SVGD')
            # plt.plot(np.log10(np.abs(meanL2normSVGDTrue_i - true_meanL2norm)), 'r.-', label='pSVGD')
            #
            # plt.legend(fontsize=16)
            # plt.xlabel("# iterations", fontsize=16)
            # plt.ylabel("Log10(error of mean)", fontsize=16)
            #
            # plt.tick_params(axis='both', which='major', labelsize=16)
            # plt.tick_params(axis='both', which='minor', labelsize=16)
            #
            # filename = "figure/error_nSamples_"+str(NSet[case])+"_mean_"+str(i)+"SVGD.pdf"
            # fig.savefig(filename, format='pdf', bbox_inches='tight')
            # filename = "figure/error_nSamples_"+str(NSet[case])+"_mean_"+str(i)+"SVGD.eps"
            # fig.savefig(filename, format='eps', bbox_inches='tight')
            #
            # plt.close()
            #
            # fig = plt.figure()
            # plt.plot(np.log10(np.abs(varianceL2normSVGDFalse_i - true_varianceL2norm)), 'bx-', label='SVGD')
            # plt.plot(np.log10(np.abs(varianceL2normSVGDTrue_i - true_varianceL2norm)), 'r.-', label='pSVGD')
            #
            # plt.legend(fontsize=16)
            # plt.xlabel("# iterations", fontsize=16)
            # plt.ylabel("Log10(error of variance)", fontsize=16)
            #
            # plt.tick_params(axis='both', which='major', labelsize=16)
            # plt.tick_params(axis='both', which='minor', labelsize=16)
            #
            # filename = "figure/error_nSamples_" + str(NSet[case]) + "_variance_" + str(i) + "SVGD.pdf"
            # fig.savefig(filename, format='pdf', bbox_inches='tight')
            # filename = "figure/error_nSamples_" + str(NSet[case]) + "_variance_" + str(i) + "SVGD.eps"
            # fig.savefig(filename, format='eps', bbox_inches='tight')
            #
            # plt.close()

    fig = plt.figure()
    plt.plot(np.log10(np.mean((meanL2normFalse_set - true_meanL2norm)**2, axis=0)), 'bx--', label='SVN, #samples=32')
    plt.plot(np.log10(np.mean((meanL2normTrue_set - true_meanL2norm)**2, axis=0)), 'r.--', label='pSVN, #samples=32')
    plt.plot(np.log10(np.mean((meanL2normFalse_set3 - true_meanL2norm)**2, axis=0)), 'bo-', label='SVN, #samples=512')
    plt.plot(np.log10(np.mean((meanL2normTrue_set3 - true_meanL2norm)**2, axis=0)), 'rs-', label='pSVN, #samples=512')

    # plt.plot(np.log10(np.mean((meanL2normSVGDFalse_set - true_meanL2norm)**2, axis=0)), 'bx--', label='SVGD, #samples=32')
    # plt.plot(np.log10(np.mean((meanL2normSVGDTrue_set - true_meanL2norm)**2, axis=0)), 'r.--', label='pSVGD, #samples=32')
    # plt.plot(np.log10(np.mean((meanL2normSVGDFalse_set3 - true_meanL2norm)**2, axis=0)), 'bo-', label='SVGD, #samples=512')
    # plt.plot(np.log10(np.mean((meanL2normSVGDTrue_set3 - true_meanL2norm)**2, axis=0)), 'rs-', label='pSVGD, #samples=512')


    plt.legend(fontsize=16)
    plt.xlabel("# iterations", fontsize=16)
    plt.ylabel("Log10(MSE of mean)", fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    filename = "figure/error_mean_MSE.pdf"
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    filename = "figure/error_mean_MSE.eps"
    fig.savefig(filename, format='eps', bbox_inches='tight')

    plt.close()

    fig = plt.figure()
    plt.plot(np.log10(np.mean((varianceL2normFalse_set - true_varianceL2norm)**2, axis=0)), 'bx--', label='SVN, #samples=32')
    plt.plot(np.log10(np.mean((varianceL2normTrue_set - true_varianceL2norm)**2, axis=0)), 'r.--', label='pSVN, #samples=32')
    plt.plot(np.log10(np.mean((varianceL2normFalse_set3 - true_varianceL2norm)**2, axis=0)), 'bo-', label='SVN, #samples=512')
    plt.plot(np.log10(np.mean((varianceL2normTrue_set3 - true_varianceL2norm)**2, axis=0)), 'rs-', label='pSVN, #samples=512')

    # plt.plot(np.log10(np.mean((varianceL2normSVGDFalse_set - true_varianceL2norm)**2, axis=0)), 'bx--', label='SVGD, #samples=32')
    # plt.plot(np.log10(np.mean((varianceL2normSVGDTrue_set - true_varianceL2norm)**2, axis=0)), 'r.--', label='pSVGD, #samples=32')
    # plt.plot(np.log10(np.mean((varianceL2normSVGDFalse_set3 - true_varianceL2norm)**2, axis=0)), 'bo-', label='SVGD, #samples=512')
    # plt.plot(np.log10(np.mean((varianceL2normSVGDTrue_set3 - true_varianceL2norm)**2, axis=0)), 'rs-', label='pSVGD, #samples=512')


    plt.legend(fontsize=16)
    plt.xlabel("# iterations", fontsize=16)
    plt.ylabel("Log10(MSE of variance)", fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    filename = "figure/error_variance_MSE.pdf"
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    filename = "figure/error_variance_MSE.eps"
    fig.savefig(filename, format='eps', bbox_inches='tight')

    plt.close()


################################## SVGD
plot = False
if plot:
    meanL2normSVGDTrue = np.zeros((Ntrial, 4))
    varianceL2normSVGDTrue = np.zeros((Ntrial, 4))
    meanL2normSVGDFalse = np.zeros((Ntrial, 4))
    varianceL2normSVGDFalse = np.zeros((Ntrial, 4))

    for case in [1,3]:

        p, N = pSet[case], NSet[case]
        if case == 1:
            meanL2normSVGDTrue_set = np.zeros((Ntrial, 100))
            varianceL2normSVGDTrue_set = np.zeros((Ntrial, 100))
            meanL2normSVGDFalse_set = np.zeros((Ntrial, 100))
            varianceL2normSVGDFalse_set = np.zeros((Ntrial, 100))

        elif case == 3:
            meanL2normSVGDTrue_set3 = np.zeros((Ntrial, 100))
            varianceL2normSVGDTrue_set3 = np.zeros((Ntrial, 100))
            meanL2normSVGDFalse_set3 = np.zeros((Ntrial, 100))
            varianceL2normSVGDFalse_set3 = np.zeros((Ntrial, 100))

        for i in range(Ntrial):

            filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
                       + str(d) + "_isProjection_" + str(True) + "_SVGD_" + str(i+1) + ".p"

            data_save = pickle.load(open(filename, 'rb'))
            meanL2normSVGDTrue_i = data_save["meanL2norm"]
            meanL2normSVGDTrue[i, case] = meanL2normFalse_i[-1]
            varianceL2normSVGDTrue_i = data_save["varianceL2norm"]
            varianceL2normSVGDTrue[i, case] = varianceL2normFalse_i[-1]

            filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
                       + str(d) + "_isProjection_" + str(False) + "_SVGD_" + str(i+1) + ".p"

            data_save = pickle.load(open(filename, 'rb'))
            meanL2normSVGDFalse_i = data_save["meanL2norm"]
            meanL2normSVGDFalse[i, case] = meanL2normFalse_i[-1]
            varianceL2normSVGDFalse_i = data_save["varianceL2norm"]
            varianceL2normSVGDFalse[i, case] = varianceL2normFalse_i[-1]

            if case == 1:
                meanL2normSVGDTrue_set[i, :] = meanL2normSVGDTrue_i
                varianceL2normSVGDTrue_set[i, :] = varianceL2normSVGDTrue_i
                meanL2normSVGDFalse_set[i, :] = meanL2normSVGDFalse_i
                varianceL2normSVGDFalse_set[i, :] = varianceL2normSVGDFalse_i

            elif case == 3:
                meanL2normSVGDTrue_set3[i, :] = meanL2normSVGDTrue_i
                varianceL2normSVGDTrue_set3[i, :] = varianceL2normSVGDTrue_i
                meanL2normSVGDFalse_set3[i, :] = meanL2normSVGDFalse_i
                varianceL2normSVGDFalse_set3[i, :] = varianceL2normSVGDFalse_i

            fig = plt.figure()
            plt.plot(np.log10(np.abs(meanL2normSVGDFalse_i - true_meanL2norm)), 'bx-', label='SVGD')
            plt.plot(np.log10(np.abs(meanL2normSVGDTrue_i - true_meanL2norm)), 'r.-', label='pSVGD')

            plt.legend(fontsize=16)
            plt.xlabel("# iterations", fontsize=16)
            plt.ylabel("Log10(error of mean)", fontsize=16)

            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=16)

            filename = "figure/error_nSamples_"+str(NSet[case])+"_mean_"+str(i)+"SVGD.pdf"
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            filename = "figure/error_nSamples_"+str(NSet[case])+"_mean_"+str(i)+"SVGD.eps"
            fig.savefig(filename, format='eps', bbox_inches='tight')

            plt.close()

            fig = plt.figure()
            plt.plot(np.log10(np.abs(varianceL2normSVGDFalse_i - true_varianceL2norm)), 'bx-', label='SVGD')
            plt.plot(np.log10(np.abs(varianceL2normSVGDTrue_i - true_varianceL2norm)), 'r.-', label='pSVGD')

            plt.legend(fontsize=16)
            plt.xlabel("# iterations", fontsize=16)
            plt.ylabel("Log10(error of variance)", fontsize=16)

            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=16)

            filename = "figure/error_nSamples_" + str(NSet[case]) + "_variance_" + str(i) + "SVGD.pdf"
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            filename = "figure/error_nSamples_" + str(NSet[case]) + "_variance_" + str(i) + "SVGD.eps"
            fig.savefig(filename, format='eps', bbox_inches='tight')

            plt.close()

    fig = plt.figure()

    plt.plot(np.log10(np.mean((meanL2normSVGDFalse_set - true_meanL2norm)**2, axis=0)), 'bx--', label='SVGD, #samples=32')
    plt.plot(np.log10(np.mean((meanL2normSVGDTrue_set - true_meanL2norm)**2, axis=0)), 'r.--', label='pSVGD, #samples=32')
    plt.plot(np.log10(np.mean((meanL2normSVGDFalse_set3 - true_meanL2norm)**2, axis=0)), 'bo-', label='SVGD, #samples=512')
    plt.plot(np.log10(np.mean((meanL2normSVGDTrue_set3 - true_meanL2norm)**2, axis=0)), 'rs-', label='pSVGD, #samples=512')

    plt.legend(fontsize=16)
    plt.xlabel("# iterations", fontsize=16)
    plt.ylabel("Log10(MSE of mean)", fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    filename = "figure/error_mean_MSE.pdf"
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    filename = "figure/error_mean_MSE.eps"
    fig.savefig(filename, format='eps', bbox_inches='tight')

    plt.close()

    fig = plt.figure()

    plt.plot(np.log10(np.mean((varianceL2normSVGDFalse_set - true_varianceL2norm)**2, axis=0)), 'bx--', label='SVGD, #samples=32')
    plt.plot(np.log10(np.mean((varianceL2normSVGDTrue_set - true_varianceL2norm)**2, axis=0)), 'r.--', label='pSVGD, #samples=32')
    plt.plot(np.log10(np.mean((varianceL2normSVGDFalse_set3 - true_varianceL2norm)**2, axis=0)), 'bo-', label='SVGD, #samples=512')
    plt.plot(np.log10(np.mean((varianceL2normSVGDTrue_set3 - true_varianceL2norm)**2, axis=0)), 'rs-', label='pSVGD, #samples=512')


    plt.legend(fontsize=16)
    plt.xlabel("# iterations", fontsize=16)
    plt.ylabel("Log10(MSE of variance)", fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    filename = "figure/error_variance_MSE.pdf"
    fig.savefig(filename, format='pdf', bbox_inches='tight')
    filename = "figure/error_variance_MSE.eps"
    fig.savefig(filename, format='eps', bbox_inches='tight')

    plt.close()







# meanL2normTrueMSE = np.mean((meanL2normTrue - true_meanL2norm)**2, axis=0)/10
# meanL2normFalseMSE = np.mean((meanL2normFalse - true_meanL2norm)**2, axis=0)/10
# varianceL2normTrueMSE = np.mean((varianceL2normTrue-true_varianceL2norm)**2, axis=0)/10
# varianceL2normFalseMSE = np.mean((varianceL2normFalse-true_varianceL2norm)**2, axis=0)/10
#
# fig = plt.figure()
#
# plt.plot(np.log10(NSet), np.log10(meanL2normTrueMSE), 'r.-', label='pSVN')
# plt.plot(np.log10(NSet), np.log10(meanL2normFalseMSE), 'bx-', label='SVN')
#
# plt.legend(fontsize=16)
# plt.xlabel("Log10(# samples)", fontsize=16)
# plt.ylabel("Log10(MSE of mean)", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=16)
#
# filename = "figure/MSE_mean.pdf"
# fig.savefig(filename, format='pdf', bbox_inches='tight')
# filename = "figure/MSE_mean.eps"
# fig.savefig(filename, format='eps', bbox_inches='tight')
#
# plt.close()
#
#
# fig = plt.figure()
#
# plt.plot(np.log10(NSet), np.log10(varianceL2normTrueMSE), 'r.-', label='pSVN')
# plt.plot(np.log10(NSet), np.log10(varianceL2normFalseMSE), 'bx-', label='SVN')
#
# plt.legend(fontsize=16)
# plt.xlabel("Log10(# samples)", fontsize=16)
# plt.ylabel("Log10(MSE of variance)", fontsize=16)
#
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=16)
#
# filename = "figure/MSE_variance.pdf"
# fig.savefig(filename, format='pdf', bbox_inches='tight')
# filename = "figure/MSE_variance.eps"
# fig.savefig(filename, format='eps', bbox_inches='tight')
#
# plt.close()
























#
# pSet = [1, 4, 16, 64]
# NSet = [8, 32, 128, 512]
# for case in range(4):
#
#     p, N = pSet[case], NSet[case]
#
#     # plots
#     fig1 = plt.figure()
#     # fig2 = plt.figure()
#     # fig3 = plt.figure()
#     # fig4 = plt.figure()
#     # fig5 = plt.figure()
#
#     for is_projection in [True, False]:
#
#         filename = "data/data_nCores_" + str(p) + "_nSamples_" + str(N) + "_nDimensions_" \
#                    + str(d) + "_isProjection_" + str(is_projection) + ".p"
#
#         data_save = pickle.load(open(filename, 'rb'))
#
#         iteration = data_save["iteration"]
#         relative_grad_norm = data_save["relative_grad_norm"]
#         relative_step_norm = data_save["relative_step_norm"]
#         meanL2norm = data_save["meanL2norm"]
#         varianceL2norm = data_save["varianceL2norm"]
#         sample_trace = data_save["sample_trace"]
#         qoi_mean = data_save["qoi_mean"]
#         qoi_std = data_save["qoi_std"]
#         pg_phat = data_save["pg_phat"]
#
#         rgn_max = []
#         rgn_mean = []
#         rsn_max = []
#         rsn_mean = []
#         pg_max = []
#         pg_mean = []
#         mln = []
#         vln = []
#         qm = []
#         qs = []
#         st = []
#         it = []
#         for i in range(len(iteration)):
#             it.append(iteration[i])
#             st.append(sample_trace[i])
#             qs.append(qoi_std[i])
#             qm.append(qoi_mean[i])
#             mln.append(meanL2norm[i])
#             vln.append(varianceL2norm[i])
#             rgn_max.append(np.max(relative_grad_norm[i]))
#             rgn_mean.append(np.mean(relative_grad_norm[i]))
#             rsn_max.append(np.max(relative_step_norm[i]))
#             rsn_mean.append(np.mean(relative_step_norm[i]))
#             pg_max.append(np.max(pg_phat[i]))
#             pg_mean.append(np.mean(pg_phat[i]))
#
#         if is_projection is True:
#             # plt.semilogy(it, np.abs(mln-true_meanL2norm)/true_meanL2norm, 'b--', label="mln_pSVN")
#             plt.semilogy(it, np.abs(vln-true_varianceL2norm)/true_varianceL2norm, 'r--', label="vln_pSVN")
#             # plt.semilogy(it, np.abs(qm-laplace_qoi_mean[4])/laplace_qoi_mean[4], 'k--', label="qm_pSVN")
#             # plt.semilogy(it, np.abs(qs-laplace_qoi_std[4])/laplace_qoi_std[4], 'g--', label="qs_pSVN")
#         else:
#             # plt.semilogy(it, np.abs(mln-true_meanL2norm)/true_meanL2norm, 'b.-', label="mln_SVN")
#             plt.semilogy(it, np.abs(vln-true_varianceL2norm)/true_varianceL2norm, 'ro-', label="vln_SVN")
#             # plt.semilogy(it, np.abs(qm-laplace_qoi_mean[4])/laplace_qoi_mean[4], 'kx-', label="qm_SVN")
#             # plt.semilogy(it, np.abs(qs-laplace_qoi_std[4])/laplace_qoi_std[4], 'g*-', label="qs_SVN")
#
#         legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
#         # legend.get_frame().set_facecolor('C0')
#
#         plt.xlabel("iterations ", fontsize=12)
#         plt.ylabel("errors", fontsize=12)
#
#         plt.tick_params(axis='both', which='major', labelsize=12)
#         plt.tick_params(axis='both', which='minor', labelsize=12)
#
#     filename = "figure/error_nSamples_"+str(N)+"_variance.pdf"
#     fig1.savefig(filename, format='pdf')
#     filename = "figure/error_nSamples_"+str(N)+"_variance.eps"
#     fig1.savefig(filename, format='eps')
#
#     plt.close()
#
#
#         # plot gradient_norm
#
#         # plot step_norm
#
#         # plot mean error
#
#         # plot variance error
#
