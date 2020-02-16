import pickle
import numpy as np
import matplotlib.pyplot as plt

sSet = [64, 128, 256, 512]
cSet = [2, 4, 8, 16]

labelSample = ['N=64', 'N=128', 'N=256', 'N=512']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']

fig = plt.figure(1)
for case in [0, 1, 2, 3]:

    s, c = sSet[case], cSet[case]

    filename = "data/data_nDimensions_1089_nCores_" + str(c) + "_nSamples_" + str(s) +"_isProjection_True_SVGD.p"

    d = pickle.load(open(filename, 'rb'))

    # print("d", d)

    plt.plot(np.log2(d["step_norm"]), makerRMSETrue[case], label=labelSample[case])

plt.xlabel("# iterations", fontsize=16)
plt.ylabel("$\log_{2}$(averaged step norm)", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/sample-stepnorm.pdf"
fig.savefig(filename, format='pdf')

plt.close()


fig = plt.figure(1)
for case in [0, 1, 2, 3]:

    s, c = sSet[case], cSet[case]

    filename = "data/data_nDimensions_1089_nCores_" + str(c) + "_nSamples_" + str(s) +"_isProjection_True_SVGD.p"

    data = pickle.load(open(filename, 'rb'))
    d = data["d_average"]
    # print("d", d)

    plt.plot(np.log10(d[0]), makerRMSETrue[case], label=labelSample[case])

plt.xlabel("$r$", fontsize=16)
plt.ylabel("$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/sample-eigenvalue.pdf"
fig.savefig(filename, format='pdf')


plt.close()
