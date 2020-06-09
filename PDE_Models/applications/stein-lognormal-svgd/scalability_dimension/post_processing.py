import pickle
import numpy as np
import matplotlib.pyplot as plt

dSet = [289, 1089, 4225, 16641]

labelTrue = ['d=289', 'd=1,089', 'd=4,225', 'd=16,641']
labelFalse = ['SVGD d=64', 'SVGD d=256', 'SVGD d=1024']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']

fig = plt.figure(1)
for case in [0, 1, 2, 3]:

    d = dSet[case]

    filename = "data/data_nDimensions_" +str(d) +"_nCores_" +str(2)+ \
               "_nSamples_" +str(64) +"_isProjection_True_SVGD.p"

    data = pickle.load(open(filename, 'rb'))
    d = data["d_average"]
    # print("d", d)

    plt.plot(np.log10(d[0]), makerRMSETrue[case], label=labelTrue[case])

plt.xlabel("$r$", fontsize=16)
plt.ylabel("$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/dimension-eigenvalue.pdf"
fig.savefig(filename, format='pdf')


plt.close()



fig = plt.figure(1)
for case in [0, 1, 2, 3]:

    d = dSet[case]

    filename = "data/data_nDimensions_" +str(d) +"_nCores_" +str(2)+ \
               "_nSamples_" +str(64) +"_isProjection_True_SVGD.p"

    data = pickle.load(open(filename, 'rb'))
    step_norm = data["step_norm"]


    plt.plot(np.log2(step_norm), makerRMSETrue[case], label=labelTrue[case])

plt.xlabel("# iterations", fontsize=16)
plt.ylabel("$\log_{2}$(averaged step norm)", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/dimension-stepnorm.pdf"
fig.savefig(filename, format='pdf')

plt.close()