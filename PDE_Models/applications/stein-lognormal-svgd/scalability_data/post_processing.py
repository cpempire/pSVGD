import pickle
import numpy as np
import matplotlib.pyplot as plt

dData = [49, 225, 961, 3969]

labelTrue = ['s=49', 's=225', 's=961', 's=3,969']
makerRMSETrue = ['gx-', 'kd-', 'mo-', 'c*-']

fig = plt.figure(1)
for case in [0, 1, 2, 3]:

    n = dData[case]

    filename = "data/d_average_nDimension_" +str(4225) +"_nCore_" +str(2)+ \
               "_nSamples_" +str(256) + "_nData_"+str(n)+'_iteration_' +str(0) +".p"

    d = pickle.load(open(filename, 'rb'))

    # print("d", d)

    plt.plot(np.log2(np.array(range(len(d)))+1), np.log10(d), makerRMSETrue[case], label=labelTrue[case])

plt.xlabel("$\log_{2}(r)$", fontsize=16)
plt.ylabel("$\log_{10}(|\lambda_r|)$", fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

filename = "figure/eigenvalue.pdf"
fig.savefig(filename, format='pdf')

plt.close()

