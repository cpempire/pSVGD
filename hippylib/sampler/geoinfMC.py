#!/usr/bin/env python
"""
Geometric Infinite dimensional MCMC samplers
Shiwei Lan @ U of Warwick, 2016
-------------------------------
After the class is instantiated with arguments, call sample to collect MCMC samples which will be stored in 'result' folder.
-----------------------------------
Created March 12, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; lanzithinking@outlook.com"

import numpy as np
import timeit,time

class geoinfMC(object):
    def __init__(self,parameter_init,Cholesky_covariance,geometry_fun,step_size,step_num,alg_name,trct_idx=[]):
        # parameters
        self.q=np.array(parameter_init)
        self.dim=len(self.q)
        self.cholC=Cholesky_covariance # Cholesky factor of (prior) covariance; cholC is lower-triangular: cholC.dot(cholC.T)=C

        # geometry needed
        self.geom=geometry_fun
        self.geom_opt=[0]
        if any(s in alg_name for s in ['MALA','HMC']): self.geom_opt.append(1)
        if any(s in alg_name for s in ['mMALA','mHMC']): self.geom_opt.append(2)
        self.nll,self.g,self.FI,self.cholG=self.geom(self.q,self.geom_opt)

        # sampling setting
        self.h=step_size
        self.L=step_num
        if 'HMC' not in alg_name: self.L=1
        self.alg_name = alg_name

        # optional setting for splitting algorithms
        self.idx_in = trct_idx # trct_idx: indices of parameters' components left
        if 'split' in alg_name:
            if not any(self.idx_in):
                self.idx_in = range(self.dim)
            self.idx_out = np.setdiff1d(range(self.dim),self.idx_in) # indices of of those components truncated out
            self.idx2_in = np.ix_(self.idx_in,self.idx_in)
            self.idx2_out = np.ix_(self.idx_out,self.idx_out)

     # preconditioned Crank-Nicolson
    def pCN(self):
        # sample velocity
        v=np.random.randn(self.dim)
        v=self.cholC.dot(v)

        # generate proposal according to Crank-Nicolson scheme
        q = ((1-self.h/4)*self.q + np.sqrt(self.h)*v)/(1+self.h/4)

        # update geometry
        nll,_,_,_=self.geom(q,self.geom_opt)

        # Metropolis test
        logr=-nll+self.nll

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # infinite dimensional Metropolis Adjusted Langevin Algorithm
    def infMALA(self):
         # sample velocity
        v=np.random.randn(self.dim)
        v=self.cholC.dot(v)

        # natural gradient
        halfng=self.cholC.T.dot(self.g)
        ng=self.cholC.dot(halfng)

        # update velocity
        v += np.sqrt(self.h)/2*ng

        # current energy
        E_cur = self.nll - np.sqrt(self.h)/2*self.g.dot(v) + self.h/8*halfng.dot(halfng)

        # generate proposal according to Langevin dynamics
        q = ((1-self.h/4)*self.q + np.sqrt(self.h)*v)/(1+self.h/4)

        # update velocity
        v = (-(1-self.h/4)*v + np.sqrt(self.h)*self.q)/(1+self.h/4)

        # update geometry
        nll,g,_,_=self.geom(q,self.geom_opt)

        # natural gradient
        halfng=self.cholC.T.dot(g)

        # new energy
        E_prp = nll - np.sqrt(self.h)/2*g.dot(v) + self.h/8*halfng.dot(halfng)

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll; self.g=g;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # infinite dimensional Hamiltonian Monte Carlo
    def infHMC(self):
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA

        # sample velocity
        v=np.random.randn(self.dim)
        v=self.cholC.dot(v)

        # natural gradient
        halfng=self.cholC.T.dot(self.g)
        ng=self.cholC.dot(halfng)

        # accumulate the power of force
        pw = rth/2*self.g.dot(v)

        # current energy
        E_cur = self.nll - self.h/8*halfng.dot(halfng)

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v+=rth/2*ng

            # a full step for position
            rot=(q+1j*v)*np.exp(-1j*rth)
            q=rot.real; v=rot.imag

            # update geometry
            nll,g,_,_=self.geom(q,self.geom_opt)
            halfng=self.cholC.T.dot(g)
            ng=self.cholC.dot(halfng)

            # another half step for velocity
            v+=rth/2*ng

            # accumulate the power of force
            if l!=randL-1: pw+=rth*g.dot(v)

        # accumulate the power of force
        pw += rth/2*g.dot(v)

        # new energy
        E_prp = nll - self.h/8*halfng.dot(halfng)

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll; self.g=g;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # infinite dimensional manifold MALA
    def infmMALA(self):
        # sample velocity
        v=np.random.randn(self.dim)
        v=np.linalg.solve(self.cholG.T,v)

        # natural gradient
        halfng=np.linalg.solve(self.cholG,self.g)
        ng=np.linalg.solve(self.cholG.T,halfng)

        # update velocity
        v += np.sqrt(self.h)/2*ng

        # current energy
        E_cur = self.nll - np.sqrt(self.h)/2*self.g.dot(v) + self.h/8*halfng.dot(halfng) +0.5*np.einsum('i,ij,j',v,self.FI,v) -sum(np.log(np.diag(self.cholG)))

        # generate proposal according to simplified manifold Langevin dynamics
        q = ((1-self.h/4)*self.q + np.sqrt(self.h)*v)/(1+self.h/4)

        # update velocity
        v = (-(1-self.h/4)*v + np.sqrt(self.h)*self.q)/(1+self.h/4)

        # update geometry
        nll,g,FI,cholG=self.geom(q,self.geom_opt)

        # natural gradient
        halfng=np.linalg.solve(cholG,g)

        # new energy
        E_prp = nll - np.sqrt(self.h)/2*g.dot(v) + self.h/8*halfng.dot(halfng) +0.5*np.einsum('i,ij,j',v,FI,v) -sum(np.log(np.diag(cholG)))

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll; self.g=g; self.FI=FI; self.cholG=cholG;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # infinite dimensional manifold HMC
    def infmHMC(self):
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA

        # sample velocity
        v=np.random.randn(self.dim)
        v=np.linalg.solve(self.cholG.T,v)

        # natural gradient
        halfng=np.linalg.solve(self.cholG,self.g)
        ng=np.linalg.solve(self.cholG.T,halfng)
        hfpricng = np.linalg.solve(self.cholC,ng) # half prior conditioned natural gradient

        # accumulate the power of force
        pw = rth/2*np.linalg.solve(self.cholC,v).dot(hfpricng)

        # current energy
        E_cur = self.nll - self.h/8*hfpricng.dot(hfpricng) +0.5*np.einsum('i,ij,j',v,self.FI,v) -sum(np.log(np.diag(self.cholG)))

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v+=rth/2*ng

            # a full step for position
#             rot=(q+1j*v)*np.exp(-1j*rth)
#             q=rot.real; v=rot.imag
            q_0=q.copy()
            q=((1-self.h/4)*q_0 + rth*v)/(1+self.h/4)
            v=((1-self.h/4)*v - rth*q_0)/(1+self.h/4)

            # update geometry
            nll,g,FI,cholG=self.geom(q,self.geom_opt)
            halfng=np.linalg.solve(cholG,g)
            ng=np.linalg.solve(cholG.T,halfng)
            hfpricng = np.linalg.solve(self.cholC,ng)

            # another half step for velocity
            v+=rth/2*ng

            # accumulate the power of force
            if l!=randL-1: pw+=rth*np.linalg.solve(self.cholC,v).dot(hfpricng)

        # accumulate the power of force
        pw += rth/2*np.linalg.solve(self.cholC,v).dot(hfpricng)

        # new energy
        E_prp = nll - self.h/8*hfpricng.dot(hfpricng) +0.5*np.einsum('i,ij,j',v,FI,v) -sum(np.log(np.diag(cholG)))

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll; self.g=g; self.FI=FI; self.cholG=cholG;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # split infinite dimensional manifold MALA
    def splitinfmMALA(self):
        # sample velocity
        v=np.random.randn(self.dim)
        v[self.idx_in] = np.linalg.solve(self.cholG.T,v[self.idx_in])
        v[self.idx_out] = self.cholC[self.idx2_out].dot(v[self.idx_out])

        # natural gradient
        halfng=np.zeros(self.dim); ng=halfng.copy()
        halfng[self.idx_in] = np.linalg.solve(self.cholG,self.g[self.idx_in]); halfng[self.idx_out] = self.cholC[self.idx2_out].T.dot(self.g[self.idx_out])
        ng[self.idx_in] = np.linalg.solve(self.cholG.T,halfng[self.idx_in]); ng[self.idx_out] = self.cholC[self.idx2_out].dot(halfng[self.idx_out])

        # update velocity
        v += np.sqrt(self.h)/2*ng

        # current energy
        E_cur = self.nll - np.sqrt(self.h)/2*self.g.dot(v) + self.h/8*halfng.dot(halfng) +0.5*np.einsum('i,ij,j',v[self.idx_in],self.FI,v[self.idx_in]) -sum(np.log(np.diag(self.cholG)))

        # generate proposal according to simplified manifold Langevin dynamics
        q = ((1-self.h/4)*self.q + np.sqrt(self.h)*v)/(1+self.h/4)

        # update velocity
        v = (-(1-self.h/4)*v + np.sqrt(self.h)*self.q)/(1+self.h/4)

        # update geometry
        nll,g,FI,cholG=self.geom(q,self.geom_opt)

        # natural gradient
        halfng[self.idx_in] = np.linalg.solve(cholG,g[self.idx_in]); halfng[self.idx_out] = self.cholC[self.idx2_out].T.dot(g[self.idx_out])

        # new energy
        E_prp = nll - np.sqrt(self.h)/2*g.dot(v) + self.h/8*halfng.dot(halfng) +0.5*np.einsum('i,ij,j',v[self.idx_in],FI,v[self.idx_in]) -sum(np.log(np.diag(cholG)))

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll; self.g=g; self.FI=FI; self.cholG=cholG;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # split infinite dimensional manifold HMC
    def splitinfmHMC(self):
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA

        # sample velocity
        v=np.random.randn(self.dim)
        v[self.idx_in] = np.linalg.solve(self.cholG.T,v[self.idx_in])
        v[self.idx_out] = self.cholC[self.idx2_out].dot(v[self.idx_out])

        # natural gradient
        halfng=np.zeros(self.dim); ng=halfng.copy()
        halfng[self.idx_in] = np.linalg.solve(self.cholG,self.g[self.idx_in]); halfng[self.idx_out] = self.cholC[self.idx2_out].T.dot(self.g[self.idx_out])
        ng[self.idx_in] = np.linalg.solve(self.cholG.T,halfng[self.idx_in]); ng[self.idx_out] = self.cholC[self.idx2_out].dot(halfng[self.idx_out])
        hfpricng = halfng # half prior conditioned natural gradient
        hfpricng[self.idx_in] = np.linalg.solve(self.cholC[self.idx2_in],ng[self.idx_in]);# hfpricng[self.idx_out] = self.cholC[self.idx2_out].T.dot(self.g[self.idx_out])

        # accumulate the power of force
        pw = rth/2*( np.linalg.solve(self.cholC[self.idx2_in],v[self.idx_in]).dot(hfpricng[self.idx_in]) + self.g[self.idx_out].dot(v[self.idx_out]) )

        # current energy
        E_cur = self.nll - self.h/8*hfpricng.dot(hfpricng) +0.5*np.einsum('i,ij,j',v[self.idx_in],self.FI,v[self.idx_in]) -sum(np.log(np.diag(self.cholG)))

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v+=rth/2*ng

            # a full step for position
#             rot=(q+1j*v)*np.exp(-1j*rth)
#             q=rot.real; v=rot.imag
            q_0=q.copy()
            q=((1-self.h/4)*q_0 + rth*v)/(1+self.h/4)
            v=((1-self.h/4)*v - rth*q_0)/(1+self.h/4)

            # update geometry
            nll,g,FI,cholG=self.geom(q,self.geom_opt)
            halfng[self.idx_in] = np.linalg.solve(cholG,g[self.idx_in]); halfng[self.idx_out] = self.cholC[self.idx2_out].T.dot(g[self.idx_out])
            ng[self.idx_in] = np.linalg.solve(cholG.T,halfng[self.idx_in]); ng[self.idx_out] = self.cholC[self.idx2_out].dot(halfng[self.idx_out])
            hfpricng = halfng # half prior conditioned natural gradient
            hfpricng[self.idx_in] = np.linalg.solve(self.cholC[self.idx2_in],ng[self.idx_in]);

            # another half step for velocity
            v+=rth/2*ng

            # accumulate the power of force
            if l!=randL-1: pw+=rth*(np.linalg.solve(self.cholC[self.idx2_in],v[self.idx_in]).dot(hfpricng[self.idx_in])+g[self.idx_out].dot(v[self.idx_out]))

        # accumulate the power of force
        pw += rth/2*(np.linalg.solve(self.cholC[self.idx2_in],v[self.idx_in]).dot(hfpricng[self.idx_in])+g[self.idx_out].dot(v[self.idx_out]))

        # new energy
        E_prp = nll - self.h/8*hfpricng.dot(hfpricng) +0.5*np.einsum('i,ij,j',v[self.idx_in],FI,v[self.idx_in]) -sum(np.log(np.diag(cholG)))

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.nll=nll; self.g=g; self.FI=FI; self.cholG=cholG;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt

    # sample with given method
    def sample(self,num_samp,num_burnin,num_retry_bad=0):
        name_sampler = str(self.alg_name)
        try:
            sampler = getattr(self, name_sampler)
        except AtributeError:
            print(self.alg_name, 'not found!')
        else:
            print('Running '+self.alg_name+' now...\n')

        # allocate space to store results
        self.samp=np.zeros((num_samp,self.dim))
        self.loglik=np.zeros(num_samp+num_burnin)
        accp=0.0 # online acceptance
        self.acpt=0.0 # final acceptance rate
        num_cons_bad=0 # number of consecutive bad proposals

        for s in range(num_samp+num_burnin):

            if s==num_burnin:
                # start the timer
                tic=timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # generate MCMC sample with given sampler
            while True:
                try:
                    acpt_idx=sampler()
                except RuntimeError:
                    if num_retry_bad==0:
                        acpt_idx=False
                        print('Bad proposal encountered! Passing... bias introduced.')
                        break # reject bad proposal: bias introduced
                    else:
                        num_cons_bad+=1
                        if num_cons_bad<num_retry_bad:
                            print('Bad proposal encountered! Retrying...')
                            continue # retry until a valid proposal is made
                        else:
                            acpt_idx=False # reject it and keep going
                            num_cons_bad=0
                            print(str(num_retry_bad)+' consecutive bad proposals encountered! Passing...')
                            break # reject it and keep going
                else:
                    num_cons_bad=0
                    break

            accp+=acpt_idx

            # display acceptance at intervals
            if (s+1)%100==0:
                print('Acceptance at %d iterations: %0.2f' % (s+1,accp/100))
                accp=0.0

            # save results
            self.loglik[s]=-self.nll
            if s>=num_burnin:
                self.samp[s-num_burnin,]=self.q.T
                self.acpt+=acpt_idx

        # stop timer
        toc=timeit.default_timer()
        self.time=toc-tic
        self.acpt/=num_samp
        print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
              % (self.time,num_samp,self.acpt))

        # save to file
        self.save()

    # save samples
    def save(self):
        import os,errno
        import pickle
        # create folder
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        try:
            os.makedirs(self.savepath)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.filename=self.alg_name+'_dim'+str(self.dim)+'_'+ctime
        # dump data
        f=open(os.path.join(self.savepath,self.filename+'.pckl'),'wb')
        pickle.dump([self.alg_name,self.h,self.L,self.samp,self.loglik,self.acpt,self.time,self.idx_in],f)
        f.close()

#         # load data
#         f=open(os.path.join(self.savepath,self.filename+'.pckl'),'rb')
#         [self.alg_name,self.h,self.L,self.samp,self.loglik,self.acpt,self.time,self.idx_in]=pickle.load(f)
#         f.close()
