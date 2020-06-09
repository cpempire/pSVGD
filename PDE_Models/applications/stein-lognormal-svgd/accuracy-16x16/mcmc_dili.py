"""
Main function to run inverse RANS model to generate posterior samples
Shiwei Lan @ Caltech, Sept. 2016
"""

from model_lognormal import *
import matplotlib.pyplot as plt
# modules
import argparse,pickle
# import dolfin as dl
# import numpy as np

# MCMC
# import sys
# sys.path.append( "../../" )
# from hippylib.sampler.DILI_hippy import DILI

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2017)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('propNO', nargs='?', type=int, default=1)
    parser.add_argument('seedNO', nargs='?', type=int, default=2017)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=10000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.15,0.25])
#     parser.add_argument('step_nums', nargs='?', type=int, default=[1])
    parser.add_argument('props', nargs='?', type=str, default=['LI_prior', 'LI_Langevin'])
    args = parser.parse_args()

    # set the (global) random seed
    np.random.seed(args.seedNO)
    print('Random seed is set to %d.' % args.seedNO)
    
    # # define the model
    # nozz_w=1.;nx=40;ny=80
    # rans=RANS(nozz_w=nozz_w,nx=nx,ny=ny)
    # rans.setup(args.seedNO,src4init='solution')
    
    # (randomly) initialize parameter
#     noise = dl.Vector()
#     rans.prior.init_vector(noise,"noise")
#     Random.normal(noise, 1., True)
#     PARAMETER=1
#     parameter = rans.model_stat.generate_vector(PARAMETER)
#     rans.whtprior.sample(noise,parameter)
#     # read from MAP
#     parameter = dl.Function(rans.Vh[PARAMETER])
#     MAP_file=os.path.join(os.getcwd(),'map_solution.h5')
#     if os.path.isfile(MAP_file):
#         f=dl.HDF5File(rans.mpi_comm,MAP_file,"r")
#         f.read(parameter,'parameter')
#         f.close()
#     else:
#         parameter=rans.get_MAP(SAVE=True)
#     parameter=rans.whtprior.u2v(parameter.vector())

    sep = "\n" + "#" * 80 + "\n"
    if rank == 0:
        print(sep, "Find the MAP point", sep)
    m = prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-8
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"] = 25
    parameters["inner_rel_tolerance"] = 1e-15
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1
    solver = ReducedSpaceNewtonCG(model, parameters)

    x = solver.solve([None, m, None])

    if rank == 0:
        if solver.converged:
            print("\nConverged in ", solver.it, " iterations.")
        else:
            print("\nNot Converged")

        print("Termination reason: ", solver.termination_reasons[solver.reason])
        print("Final gradient norm: ", solver.final_grad_norm)
        print("Final cost: ", solver.final_cost)

    # run MCMC to generate samples
    adpt_stepsz=True

    # if rans.rank == 0:
    #     print(str("Preparing DILI sampler using {} proposal with "+{True:"initial",False:"fixed"}[adpt_stepsz]+" step sizes {} for LIS and CS resp....").format(args.props[args.propNO],args.step_sizes,))

    dili=DILI(x[PARAMETER], model, args.step_sizes, args.props[args.propNO], adpt_stepsz, target_acpt=0.7, n_lag=100)
    dili.setup(args.num_samp,args.num_burnin,1)
    dili.adaptive_MCMC(num_retry_bad=0,threshold_l=1e-3,threshold_g=1e-3)

    iact, lags, acoors = integratedAutocorrelationTime(dili.data[:, 1])

    print("iact, lags, acoors", iact, lags, acoors)
    print("Integrated autocorrelation time = {0}".format(iact))

    fig = plt.figure()
    plt.plot(lags, acoors, '.-')
    plt.xlabel("lag", fontsize=16)
    plt.ylabel("autocorr", fontsize=16)
    plt.show()
    filename = "figure/mcmc_autocorr.pdf"
    fig.savefig(filename, format='pdf')
    filename = "figure/mcmc_autocorr.eps"
    fig.savefig(filename, format='eps')

    pickle.dump(dili.data, open("data/mcmc_dili.p", 'wb'))
    print("E[qoi] = {0}".format(np.mean(dili.data[:, 0])))
    print("Std[qoi] = {0}".format(np.std(dili.data[:, 0])))
    print("E[cost] = {0}".format(np.mean(dili.data[:, 1])))
    print("Std[cost] = {0}".format(np.std(dili.data[:, 1])))

    data = dict()
    sample_mean = dili.m.get_local()
    sample_2nd_moment = dili.m2.get_local()
    data["mean"] = sample_mean
    data["variance"] = sample_2nd_moment - sample_mean**2
    data["sample_2nd_moment"] = sample_2nd_moment
    pickle.dump(data, open("data/mcmc_dili_sample.p", 'wb'))

    fig = plt.figure()
    plt.plot(dili.data[:, 0], 'b.')
    plt.show()
    filename = "figure/mcmc_dili.pdf"
    fig.savefig(filename, format='pdf')
    filename = "figure/mcmc_dili.eps"
    fig.savefig(filename, format='eps')

#     # rename
#     filename_=os.path.join(dili.savepath,dili.filename+'.pckl')
#     filename=os.path.join(dili.savepath,'RANS_seed'+str(args.seedNO)+'_'+dili.filename+'.pckl') # change filename
#     if rans.rank == 0:
#         os.rename(filename_, filename)
#     # append PDE information including the count of solving
#     f=open(filename,'ab')
# #     soln_count=rans.soln_count.copy()
#     soln_count=[rans.pde.solveFwd.count,rans.pde.solveAdj.count,rans.pde.solveIncremental.count]
#     pickle.dump([nozz_w,nx,ny,fresh_init,para_cont,soln_count,args],f)
#     f.close()
#     print('PDE solution counts (forward, adjoint, incremental): {} \n'.format(soln_count))
# #     # verify with load
# #     f=open(filename,'rb')
# #     mc_samp=pickle.load(f)
# #     pde_info=pickle.load(f)
# #     f.close
# #     print(pde_cnt)

if __name__ == '__main__':
    main()
