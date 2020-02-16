import pickle
import numpy as np
import matplotlib.pyplot as plt

time_communication = []
time_computation = []
n = 6
particle = np.zeros((n, 2))
variation = np.zeros((n, 2))
kernel = np.zeros((n, 2))
solve = np.zeros((n, 2))
total = np.zeros(n)

n = 0
for p in [1, 2, 4, 8, 16, 32]:
    filename = "data/time_nCore_"+str(p)+'.p'
    data_save = pickle.load(open(filename,'rb'))
    particle[n, :] = [data_save["time_communication"][0], data_save["time_computation"][0]]
    variation[n, :] = [data_save["time_communication"][1], data_save["time_computation"][1]]
    kernel[n, :] = [data_save["time_communication"][2], data_save["time_computation"][2]]
    solve[n, :] = [data_save["time_communication"][3], data_save["time_computation"][3]]
    total[n] = data_save["time_total"]

    n += 1

processor = [1, 2, 4, 8, 16, 32]

speedup_total = total[0]/total
speedup_sample = particle[0,1]/particle[:,1]
speedup_variation = variation[0,1]/variation[:,1]
speedup_kernel = kernel[0,1]/kernel[:,1]
speedup_solve = solve[0,1]/solve[:,1]

fig, ax = plt.subplots()
ax.set_xscale('symlog', basex=2)
ax.set_yscale('symlog', basey=2)
ax.plot(processor, speedup_total, 'r.-', label="total")
ax.plot(processor, speedup_variation, 'ko-', label="variation")
ax.plot(processor, speedup_kernel, 'g*-', label="kernel")
ax.plot(processor, speedup_solve, 'ms-', label="update")
ax.plot(processor, speedup_sample, 'bx-', label="sample")
ax.plot(processor, processor, 'k--', label="$N$")
plt.legend()
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.xlabel("# processor cores", fontsize=16)
plt.ylabel("speedup", fontsize=16)

filename = "figure/core_speedup.pdf"
fig.savefig(filename, format='pdf')
filename = "figure/core_speedup.eps"
fig.savefig(filename, format='eps')

plt.close()

fig, ax = plt.subplots()
# ax.set_xscale('symlog', basex=2)
# ax.set_yscale('symlog', basey=2)
ax.plot(np.log2(processor[:]), np.log2(total[:]), 'r.-', label="total")
ax.plot(np.log2(processor[:]), np.log2(variation[:,1]), 'ko-', label="gradient")
ax.plot(np.log2(processor[:]), np.log2(kernel[:,1]), 'g*-', label="kernel")
ax.plot(np.log2(processor[:]), np.log2(solve[:,1]), 'ms-', label="update")
# ax.plot(np.log2(processor), np.log2(particle[:,1]), 'bx-', label="sample")
ax.plot([0,5], [11,6], '--', label="$O(K^{-1})$")
plt.legend(fontsize=14)
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.xlabel("$\log_2$(# processor cores)", fontsize=16)
plt.ylabel("$\log_2$(time (s))", fontsize=16)

filename = "figure/core_time.pdf"
fig.savefig(filename, format='pdf')
filename = "figure/core_time.eps"
fig.savefig(filename, format='eps')

plt.close()


# speedup = total[0]/total
#
# fig, ax = plt.subplots()
# ax.set_xscale('symlog', basex=2)
# ax.set_yscale('symlog', basey=2)
# ax.plot(processor, speedup, 'r.-', label="total")
# ax.plot(processor, processor, 'k--', label="$N$")
# plt.legend()
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))
#
# plt.xlabel("cores", fontsize=12)
# plt.ylabel("speed up", fontsize=12)
#
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=12)
#
# filename = "figure/core_speedup_total.pdf"
# fig.savefig(filename, format='pdf')
# filename = "figure/core_speedup_total.eps"
# fig.savefig(filename, format='eps')
#
# plt.close()
#
#
# speedup = particle[0, 1]/particle[:, 1]
#
# fig, ax = plt.subplots()
# ax.set_xscale('symlog', basex=2)
# ax.set_yscale('symlog', basey=2)
# ax.plot(processor, speedup, 'r.-', label="sample")
# ax.plot(processor, processor, 'k--', label="$N$")
# plt.legend()
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))
#
# plt.xlabel("cores", fontsize=12)
# plt.ylabel("speed up", fontsize=12)
#
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=12)
#
# filename = "figure/core_speedup_particle.pdf"
# fig.savefig(filename, format='pdf')
# filename = "figure/core_speedup_particle.eps"
# fig.savefig(filename, format='eps')
#
# plt.close()
#
#
# speedup = variation[0, 1]/variation[:, 1]
#
# fig, ax = plt.subplots()
# ax.set_xscale('symlog', basex=2)
# ax.set_yscale('symlog', basey=2)
# ax.plot(processor, speedup, 'r.-', label="variation")
# ax.plot(processor, processor, 'k--', label="$N$")
# plt.legend()
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))
#
# plt.xlabel("cores", fontsize=12)
# plt.ylabel("speed up", fontsize=12)
#
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=12)
#
# filename = "figure/core_speedup_variation.pdf"
# fig.savefig(filename, format='pdf')
# filename = "figure/core_speedup_variation.eps"
# fig.savefig(filename, format='eps')
#
# plt.close()
#
#
# speedup = kernel[0, 1]/kernel[:, 1]
#
# fig, ax = plt.subplots()
# ax.set_xscale('symlog', basex=2)
# ax.set_yscale('symlog', basey=2)
# ax.plot(processor, speedup, 'r.-', label="kernel")
# ax.plot(processor, processor, 'k--', label="$N$")
# plt.legend()
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))
#
# plt.xlabel("cores", fontsize=12)
# plt.ylabel("speed up", fontsize=12)
#
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=12)
#
# filename = "figure/core_speedup_kernel.pdf"
# fig.savefig(filename, format='pdf')
# filename = "figure/core_speedup_kernel.eps"
# fig.savefig(filename, format='eps')
#
# plt.close()
#
#
# speedup = solve[0, 1]/solve[:, 1]
#
# fig, ax = plt.subplots()
# ax.set_xscale('symlog', basex=2)
# ax.set_yscale('symlog', basey=2)
# ax.plot(processor, speedup, 'r.-', label="solve")
# ax.plot(processor, processor, 'k--', label="$N$")
# plt.legend()
# plt.xlim((processor[0],processor[-1]))
# plt.ylim((processor[0],processor[-1]))
#
# plt.xlabel("cores", fontsize=12)
# plt.ylabel("speed up", fontsize=12)
#
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.tick_params(axis='both', which='minor', labelsize=12)
#
# filename = "figure/core_speedup_solve.pdf"
# fig.savefig(filename, format='pdf')
# filename = "figure/core_speedup_solve.eps"
# fig.savefig(filename, format='eps')
#
# plt.close()
