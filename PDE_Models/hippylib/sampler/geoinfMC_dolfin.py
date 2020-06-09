#!/usr/bin/env python
"""
Geometric Infinite dimensional MCMC samplers
using FEniCS dolfin library https://bitbucket.org/fenics-project/dolfin
Shiwei Lan @ Caltech, 2016
-------------------------------
This is to accompany this paper: https://arxiv.org/abs/1606.06351
It has been configured to use low-rank Hessian approximation.
After the class is instantiated with arguments, call sample to collect MCMC samples which will be stored in 'result' folder.
-----------------------------------
Created October 11, 2016
Modified February 26, 2017
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@outlook.com"

import numpy as np
import timeit,time
import dolfin as df

class geoinfMC:
    """
    Geometric Infinite dimensional MCMC samplers by Beskos, Alexandros, Mark Girolami, Shiwei Lan, Patrick E. Farrell, and Andrew M. Stuart.
    https://www.sciencedirect.com/science/article/pii/S0021999116307033
    -------------------------------------------------------------------
    It has been configured to use low-rank Hessian approximation.
    After the class is instantiated with arguments, call sample to collect MCMC samples which will be stored in 'result' folder.
    """
    def __init__(self,parameter_init,model,step_size,step_num,alg_name,adpt_h=False,**kwargs):
        """
        Initialization
        """
        # parameters
        self.q=parameter_init
        self.dim=self.q.size()
        self.model=model
        
        target_acpt=kwargs.pop('target_acpt',0.65)
        # geometry needed
        geom_ord=[0]
        if any(s in alg_name for s in ['MALA','HMC']): geom_ord.append(1)
        if any(s in alg_name for s in ['mMALA','mHMC']): geom_ord.append(2)
        self.geom=lambda parameter: model.get_geom(parameter,geom_ord=geom_ord,**kwargs)
        self.ll,self.g,_,self.eigs=self.geom(self.q)

        # sampling setting
        self.h=step_size
        self.L=step_num
        if 'HMC' not in alg_name: self.L=1
        self.alg_name = alg_name

        # optional setting for adapting step size
        self.adpt_h=adpt_h
        if self.adpt_h:
            h_adpt={}
#             h_adpt['h']=self._init_h()
            h_adpt['h']=self.h
            h_adpt['mu']=np.log(10*h_adpt['h'])
            h_adpt['loghn']=0.
            h_adpt['An']=0.
            h_adpt['gamma']=0.05
            h_adpt['n0']=10
            h_adpt['kappa']=0.75
            h_adpt['a0']=target_acpt
            self.h_adpt=h_adpt
    
    def randv(self,post_Ga=None):
        """
        sample v ~ N(0,C) or N(0,invK(q))
        """
        if post_Ga is None:
            v = self.model.prior.sample()
        else:
            v = self.model.post_Ga.sample()
        return v
        
    def pCN(self):
        """
        preconditioned Crank-Nicolson
        """
        # initialization
        q=self.q.copy()
        
        # sample velocity
        v=self.randv()

        # generate proposal according to Crank-Nicolson scheme
        q.zero()
        q.axpy((1-self.h/4)/(1+self.h/4),self.q)
        q.axpy(np.sqrt(self.h)/(1+self.h/4),v)

        # update geometry
        ll,_,_,_=self.geom(q)

        # Metropolis test
        logr=ll-self.ll

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def infMALA(self):
        """
        infinite dimensional Metropolis Adjusted Langevin Algorithm
        """
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h)
        
        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.model.prior.C_act(self.g)

        # update velocity
        v.axpy(rth/2,ng)

        # current energy
        E_cur = -self.ll - rth/2*self.g.inner(v) + self.h/8*self.g.inner(ng)

        # generate proposal according to Langevin dynamics
        q.zero()
        q.axpy((1-self.h/4)/(1+self.h/4),self.q)
        q.axpy(rth/(1+self.h/4),v)

        # update velocity
        v_=v.copy();v.zero()
        v.axpy(-(1-self.h/4)/(1+self.h/4),v_)
        v.axpy(rth/(1+self.h/4),self.q)

        # update geometry
        ll,g,_,_=self.geom(q)

        # natural gradient
        ng=self.model.prior.C_act(g)

        # new energy
        E_prp = -ll - rth/2*g.inner(v) + self.h/8*g.inner(ng)

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def infHMC(self):
        """
        infinite dimensional Hamiltonian Monte Carlo
        """
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
        cos_=np.cos(rth); sin_=np.sin(rth);

        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.model.prior.C_act(self.g)

        # accumulate the power of force
        pw = rth/2*self.g.inner(v)

        # current energy
        E_cur = -self.ll - self.h/8*self.g.inner(ng)

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v.axpy(rth/2,ng)

            # a full step for position
            q_=q.copy();q.zero()
            q.axpy(cos_,q_)
            q.axpy(sin_,v)
            v_=v.copy();v.zero()
            v.axpy(-sin_,q_)
            v.axpy(cos_,v_)

            # update geometry
            ll,g,_,_=self.geom(q)
            ng=self.model.prior.C_act(g)

            # another half step for velocity
            v.axpy(rth/2,ng)

            # accumulate the power of force
            if l!=randL-1: pw+=rth*g.inner(v)

        # accumulate the power of force
        pw += rth/2*g.inner(v)

        # new energy
        E_prp = -ll - self.h/8*g.inner(ng)

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def DRinfmMALA(self):
        """
        dimension-reduced infinite dimensional manifold MALA
        """
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h)
        
        # sample velocity
        v=self.randv(self.model.post_Ga)

        # natural gradient
        ng=self.model.post_Ga.postC_act(self.g) # use low-rank posterior Hessian solver

        # update velocity
        v.axpy(rth/2,ng)

        # current energy
        E_cur = -self.ll - rth/2*self.g.inner(v) + self.h/8*self.g.inner(ng) +0.5*self.model.post_Ga.Hlr.norm2(v) -0.5*sum(np.log(1+self.eigs[0])) # use low-rank Hessian inner product

        # generate proposal according to simplified manifold Langevin dynamics
        q.zero()
        q.axpy((1-self.h/4)/(1+self.h/4),self.q)
        q.axpy(rth/(1+self.h/4),v)

        # update velocity
        v_=v.copy();v.zero()
        v.axpy(-(1-self.h/4)/(1+self.h/4),v_)
        v.axpy(rth/(1+self.h/4),self.q)

        # update geometry
        ll,g,_,eigs=self.geom(q)
        self.model.post_Ga.eigs=eigs # update the eigen-pairs in low-rank approximation --important!

        # natural gradient
        ng=self.model.post_Ga.postC_act(g)

        # new energy
        E_prp = -ll - rth/2*g.inner(v) + self.h/8*g.inner(ng) +0.5*self.model.post_Ga.Hlr.norm2(v) -0.5*sum(np.log(1+eigs[0]))

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g; self.eigs=eigs;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def DRinfmHMC(self):
        """
        dimension-reduced infinite dimensional manifold HMC
        """
        # initialization
        q=self.q.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
#         cos_=np.cos(rth); sin_=np.sin(rth);
        cos_=(1-self.h/4)/(1+self.h/4); sin_=rth/(1+self.h/4);

        # sample velocity
        v=self.randv(self.model.post_Ga)

        # natural gradient
        ng=self.model.post_Ga.postC_act(self.g) # use low-rank posterior Hessian solver

        # accumulate the power of force
        pw = rth/2*self.model.prior.C_act(v,-1).inner(ng)

        # current energy
        E_cur = -self.ll + self.h/4*self.model.prior.logpdf(ng) +0.5*self.model.post_Ga.Hlr.norm2(v) -0.5*sum(np.log(1+self.eigs[0])) # use low-rank Hessian inner product

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v.axpy(rth/2,ng)

            # a full step rotation
            q_=q.copy();q.zero()
            q.axpy(cos_,q_)
            q.axpy(sin_,v)
            v_=v.copy();v.zero()
            v.axpy(cos_,v_)
            v.axpy(-sin_,q_)

            # update geometry
            ll,g,_,eigs=self.geom(q)
            self.model.post_Ga.eigs=eigs # update the eigen-pairs in low-rank approximation --important!
            ng=self.model.post_Ga.postC_act(g)

            # another half step for velocity
            v.axpy(rth/2,ng)

            # accumulate the power of force
            if l!=randL-1: pw+=rth*self.model.prior.C_act(v,-1).inner(ng)

        # accumulate the power of force
        pw += rth/2*self.model.prior.C_act(v,-1).inner(ng)

        # new energy
        E_prp = -ll + self.h/4*self.model.prior.logpdf(ng) +0.5*self.model.post_Ga.Hlr.norm2(v) -0.5*sum(np.log(1+eigs[0]))

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.q=q; self.ll=ll; self.g=g; self.eigs=eigs;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr
    
    def _init_h(self):
        """
        find a reasonable initial step size
        """
        h=1.
        _self=self
        sampler=getattr(_self,str(_self.alg_name))
        _self.h=h;_self.L=1
        _,logr=sampler()
        a=2.*(np.exp(logr)>0.5)-1.
        while a*logr>-a*np.log(2):
            h*=pow(2.,a)
            _self=self
            _self.h=h;_self.L=1
            _,logr=sampler()
        return h
    
    def _dual_avg(self,iter,an):
        """
        dual-averaging to adapt step size
        """
        hn_adpt=self.h_adpt
        hn_adpt['An']=(1.-1./(iter+hn_adpt['n0']))*hn_adpt['An'] + (hn_adpt['a0']-an)/(iter+hn_adpt['n0'])
        logh=hn_adpt['mu'] - np.sqrt(iter)/hn_adpt['gamma']*hn_adpt['An']
        hn_adpt['loghn']=pow(iter,-hn_adpt['kappa'])*logh + (1.-pow(iter,-hn_adpt['kappa']))*hn_adpt['loghn']
        hn_adpt['h']=np.exp(logh)
        return hn_adpt
    
    # sample with given method
    def sample(self,num_samp,num_burnin,num_retry_bad=0,**kwargs):
        """
        sample with given MCMC method
        """
        name_sampler = str(self.alg_name)
        try:
            sampler = getattr(self, name_sampler)
        except AtributeError:
            print(self.alg_name, 'not found!')
        else:
            print('\nRunning '+self.alg_name+' now...\n')

        # allocate space to store results
        import os
        samp_fname='_samp_'+self.alg_name+'_dim'+str(self.dim)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")
        samp_fpath=os.path.join(os.getcwd(),'result')
        if not os.path.exists(samp_fpath):
            os.makedirs(samp_fpath)
#         self.samp=df.File(os.path.join(samp_fpath,samp_fname+".xdmf"))
        self.samp=df.HDF5File(self.model.pde.mpi_comm,os.path.join(samp_fpath,samp_fname+".h5"),"w")
        self.loglik=np.zeros(num_samp+num_burnin)
        self.acpt=0.0 # final acceptance rate
        self.times=np.zeros(num_samp+num_burnin) # record the history of time used for each sample
        
        # number of adaptations for step size
        if self.adpt_h:
            self.h_adpt['n_adpt']=kwargs.pop('adpt_steps',num_burnin)
        
        # online parameters
        accp=0.0 # online acceptance
        num_cons_bad=0 # number of consecutive bad proposals

        beginning=timeit.default_timer()
        for s in xrange(num_samp+num_burnin):

            if s==num_burnin:
                # start the timer
                tic=timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # generate MCMC sample with given sampler
            while True:
                try:
                    acpt_idx,logr=sampler()
                except RuntimeError as e:
                    print(e)
                    if num_retry_bad==0:
                        acpt_idx=False; logr=-np.inf
                        print('Bad proposal encountered! Passing... bias introduced.')
                        break # reject bad proposal: bias introduced
                    else:
                        num_cons_bad+=1
                        if num_cons_bad<num_retry_bad:
                            print('Bad proposal encountered! Retrying...')
                            continue # retry until a valid proposal is made
                        else:
                            acpt_idx=False; logr=-np.inf # reject it and keep going
                            num_cons_bad=0
                            print(str(num_retry_bad)+' consecutive bad proposals encountered! Passing...')
                            break # reject it and keep going
                else:
                    num_cons_bad=0
                    break

            accp+=acpt_idx

            # display acceptance at intervals
            if (s+1)%100==0:
                print('\nAcceptance at %d iterations: %0.2f' % (s+1,accp/100))
                accp=0.0

            # save results
            self.loglik[s]=self.ll
            if s>=num_burnin:
#                 self.samp << vec2fun(self.q,self.model.prior.V,name='sample_{0}'.format(s-num_burnin))
                q_f=df.Function(self.model.prior.V)
                q_f.vector()[:]=self.q
                self.samp.write(q_f,'sample_{0}'.format(s-num_burnin))
                self.acpt+=acpt_idx
            
            # record the time
            self.times[s]=timeit.default_timer()-beginning
            
            # adapt step size h if needed
            if self.adpt_h:
                if s<self.h_adpt['n_adpt']:
                    self.h_adpt=self._dual_avg(s+1,np.exp(min(0,logr)))
                    self.h=self.h_adpt['h']
                    print('New step size: %.2f; \t New averaged step size: %.6f\n' %(self.h_adpt['h'],np.exp(self.h_adpt['loghn'])))
                if s==self.h_adpt['n_adpt']:
                    self.h_adpt['h']=np.exp(self.h_adpt['loghn'])
                    self.h=self.h_adpt['h']
                    print('Adaptation completed; step size freezed at:  %.6f\n' % self.h_adpt['h'])

        # stop timer
        self.samp.close()
        toc=timeit.default_timer()
        self.time=toc-tic
        self.acpt/=num_samp
        print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
              % (self.time,num_samp,self.acpt))

        # save to file
        self.save_samp()

    # save samples
    def save_samp(self):
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
        res2save=[self.h,self.L,self.alg_name,self.loglik,self.acpt,self.time,self.times]
        if self.adpt_h:
            res2save.append(self.h_adpt)
        pickle.dump(res2save,f)
        f.close()
#         # load data
#         f=open(os.path.join(self.savepath,self.filename+'.pckl'),'rb')
#         f_read=pickle.load(f)
#         f.close()
