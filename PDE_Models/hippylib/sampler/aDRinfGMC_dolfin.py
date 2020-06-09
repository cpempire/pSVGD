#!/usr/bin/env python
"""
Adaptive Dimension-Reduced infinite-dimensional manifold MCMC
modified from:
------------------------------------------------------------------
Dimension-independent likelihood-informed (DILI) MCMC
by Tiangang Cui, Kody J.H. Law, and Youssef M. Marzouk
http://www.sciencedirect.com/science/article/pii/S0021999115006701
------------------------------------------------------------------
Created March 23, 2017
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017, The EQUiPS project"
__license__ = "GPL"
__version__ = "0.8"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@outlook.com"

import dolfin as df
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
import timeit,time
import sys
sys.path.append( "../" )
from util import Eigen,wtQR

# def CholQR(Y,W):
#     """
#     CholQR with W-inner products
#     ----------------------------
#     Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
#     Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion,
#     Numerical Linear Algebra with Applications 23 (2), pp. 314-339.
#     """
#     Z=W.dot(Y) if type(W) is np.ndarray else np.array([W(r) for r in Y.T]).T
#     C=Y.T.dot(Z)
#     L=np.linalg.cholesky(C)
#     Q=np.linalg.solve(L,Y.T).T
#     return Q,L.T

class aDRinfGMC:
    """
    Adaptive Dimension-Reduced infinite-dimensional manifold MCMC, modified from:
    Dimension-independent likelihood-informed (DILI) MCMC by Tiangang Cui, Kody J.H. Law, and Youssef M. Marzouk
    http://www.sciencedirect.com/science/article/pii/S0021999115006701
    ------------------------------------------------------------------
    The main purpose is to speed up the mixing of dimension-independent MCMC defined on function (Hilbert) space.
    The key idea is to find likelihood-informed (low dimensional) subspace (LIS) using prior pre-conditioned Gauss-Newton approximation of Hessian (ppGNH) averaged wrt. posterior samples;
    and apply more sophisticated/expensive methods (Langevin) in LIS and explore the complement subspace (CS) with more efficient/cheap methods like pCN.
    ------------------------------------------------------------------
    After the class is instantiated with arguments, call adaptive_MCMC to collect MCMC samples which will be stored in 'result' folder.
    """
    def __init__(self,parameter_init,model,step_size,step_num=1,alg_name='aDRinfmMALA',prop_name='LI_Langevin',adpt_h=False,**kwargs):
        """
        Initialize adaptive dimension-reduced infinite-dimensional manifold MCMC instance with parameter, function providing geometric information, step size and proposal method:
        v                 : whitened parameter (vector) to sample
        model             : model to provide geometric information including log-density (likelihood), its gradient and Hessian (or Fisher) etc.
        Lambda_m,Nu_m     : accumulated LIS basis-- eigenvalues and eigenvectors (eigenfunctions)
        Lambda_r,Nu_r     : global LIS basis
        d_F,treshhold_LIS : Forstner distance between two covariances to diagnose the convergence of LIS if it drops below treshhold_LIS
        h                 : step size(s) of MCMC
        L                 : number of leap-frog steps
        n_lag,n_max       : interval/maximum number of updating LIS
        alg_name          : name of MCMC algorithm
        prop_name         : option for proposal in MCMC
        adpt_h            : indicator to adapt step size(s)
        """
        # parameter
        self.v=parameter_init
        self.dim=parameter_init.size()
        # model
        self.model=model
        
        # sampling setting
        self.h=step_size
        self.L=step_num
        self.n_lag=kwargs.pop('n_lag',200)
        self.n_max=kwargs.pop('n_max',1000)
        self.threshold_LIS=kwargs.pop('threshold_LIS',1e-5)
        
        target_acpt=kwargs.pop('target_acpt',0.65)
        # geometry needed
        self.geom=lambda parameter,geom_ord=[0],**kwargs: model.get_geom(parameter,geom_ord=geom_ord,whitened=True,**kwargs)
        self.loglik,self.gradlik,_,self.eigs=self.geom(self.v,geom_ord=[0,1,1.5],threshold=0.01)
        
        # LIS basis
        self.update_LIS_m=0
        self.Lambda_m,self.Nu_m=self.eigs # local
        self.dim_LIS_l=self.Lambda_m.size
        self.Lambda_r,self.Nu_r=self.eigs # global
        self.dim_LIS_g=self.Lambda_r.size
        print('Initial local/global LIS has %d dimensions.' % self.dim_LIS_l)
        self.d_F=np.inf # Forstner distance to detect convergence of LIS
        # initialize re-weighted basis
        self.D_r=1./(1+self.Lambda_r)
        
        # algorithm
        self.alg_name=alg_name
        # operators
        self.prop_name=prop_name
        self.form_operators()
        
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
    
    def form_operators(self):
        """
        Form operators A, B, and G induced by reweighted LIS basis
        Input: Nu_r, D_r, (h)
        Output(aDRinfmMALA): D_Ar, D_Br, D_Gr, A, B, G
        Output(aDRinfmHMC): Nu_v, Nuv
        """
        if self.alg_name is 'aDRinfmMALA':
            rho0=(4.-self.h)/(4.+self.h)
            rho1=1-rho0; rho2=np.sqrt(1-rho0**2)
            # on LIS
            self.D_Ar=1-rho1*self.D_r
            self.D_Br=rho2*np.sqrt(self.D_r)
            if self.prop_name is 'LI_prior':
                self.D_Gr=np.zeros_like(self.D_r)
            elif self.prop_name is 'LI_Langevin':
                self.D_Gr=rho1*self.D_r
            else:
                print('Wrong proposal!')
                raise
            # on the complement space
            a_perp=rho0
            b_perp=rho2
            # operators
            self.A=lambda x:self.Nu_r.dot((self.D_Ar-a_perp)*self.Nu_r.T.dot(self.model.prior.M*x)) + a_perp*x
            self.B=lambda x:self.Nu_r.dot((self.D_Br-b_perp)*self.Nu_r.T.dot(self.model.prior.M*x)) + b_perp*x
            self.G=lambda x:self.Nu_r.dot(self.D_Gr*self.Nu_r.T.dot(x)) # G only applies to an assembled gradient
        elif self.alg_name is 'aDRinfmHMC':
            # operators
            self.Nu_v=lambda x:self.Nu_r.T.dot(self.model.prior.M*x)
#             self.rtK=lambda x:self.Nu_r.dot((np.sqrt(self.D_r)-1.)*self.Nu_r.T.dot(self.model.prior.M*x)) + x
#             self.rtK=lambda x:sefl.model.prior.gen_vector(self.Nu_r.dot((np.sqrt(self.D_r)-1.)*self.Nu_v(x)) + x)
#             self._g_r=lambda x,g:self.Lambda_r*self.Nu_v(x)+{'LI_prior':0,'LI_Langevin':self.Nu_r.T.dot(g)}[self.prop_name] # g is already assembled
            self.Nuv=lambda x:self.model.prior.gen_vector(self.Nu_r.dot(x))
        else:
            print('Algorithm not available!')
            raise
    
    def rand_aux(self):
        """
        Generate random vector: _v ~ N(0,_K), _K = I + Nu_r * (D_r-I_r) * Nu_r'
        """
        _v=self.model.prior.sample(whiten=True)
        _v.axpy(1.,self.model.prior.gen_vector(self.Nu_r.dot((np.sqrt(self.D_r)-1.)*self.Nu_v(_v))))
        return _v
    
    def _g_r(self,v,g=None):
        """
        Operator g_r: v --> Lambda_r * Nu_r' * v - gamma_r * Nu_r' * D_v Phi
        return the gradient projected to subspace
        """
        g_r=self.Lambda_r*self.Nu_v(v)
        if self.prop_name is 'LI_prior':
            return g_r
        elif self.prop_name is 'LI_Langevin':
            g_r+=self.Nu_r.T.dot(g) # g is already assembled
            return g_r
        else:
            print('Wrong proposal!')
            raise
    
    def update_LIS(self,threshold_l=.01,threshold_s=1e-4,threshold_g=.01):
        """
        Algorithm 1: Incremental update of the expected GNH and global LIS.
        Input: Nu_m, Lambda_m, v_m1
        Output: Nu_m1, Lambda_m1, Nu_r, Lambda_r
        """
        # count the number of calls
        self.update_LIS_m+=1
        m=self.update_LIS_m
        # Compute the local LIS basis
        _,_,_,eigs=self.geom(self.v,geom_ord=[1.5],threshold=threshold_l)
        Lambda_m1,Phi_m1=eigs
        self.dim_LIS_l=Lambda_m1.size
        # Compute the QR decomposition
#         Q,R=np.linalg.qr(np.hstack([self.Nu_m,Phi_m1]))
        Q,R=wtQR.CholQR(np.hstack([self.Nu_m,Phi_m1]),lambda x:self.model.prior.M*x) # QR decomposition in weighted space R_M
        # Compute the new eigenvalues through the eigendecomposition
        eig_aug=np.hstack([m*self.Lambda_m,Lambda_m1])
        mat_aug=(R*eig_aug).dot(R.T)/(m+1)
#         k=min([eig_aug.shape[0],mat_aug.shape[0]-1])
#         Lambda_m1,W=sps.linalg.eigsh(mat_aug,k=k)
        Lambda_m1,W=np.linalg.eigh(mat_aug)
        dsc_ord = Lambda_m1.argsort()[::-1]
        Lambda_m1 = Lambda_m1[dsc_ord]; W = W[:,dsc_ord]
        # Compute the new basis
        Nu_m1=Q.dot(W)
        # truncate eigenvalues for updated LIS
        idx_s=Lambda_m1>=threshold_s
        self.Nu_m=Nu_m1[:,idx_s]
        self.Lambda_m=Lambda_m1[idx_s]
        # truncate eigenvalues for global LIS to return
        idx_g=Lambda_m1>=threshold_g
        self.Nu_r=Nu_m1[:,idx_g]
        self.Lambda_r=Lambda_m1[idx_g]
        self.dim_LIS_g=self.Lambda_r.size
#         return self.Nu_r,self.Lambda_r
    
    def aDRinfmMALA(self,v_ref=None):
        """
        aDRinfmMALA kernel to generate a (posterior) sample.
        Input: v, [loglik, gradlik]
        Output: v1, [loglik1, gradlik1], acpt (indicator)
        """
#         if v_ref is None:
#             v_ref=np.zeros_like(self.v)
        # Compute a candidate using either LI-prior or LI-Langevin
        # Compute the (log) acceptance probability
        xi=self.model.prior.sample(whiten=True)
        v=self.A(self.v)+self.B(xi)
        log_acpt=0 # log of acceptance probability
        if self.prop_name is 'LI_prior':
            loglik,_,_,_=self.geom(v)
        elif self.prop_name is 'LI_Langevin':
            v+=self.G(self.gradlik)
            v=self.model.prior.gen_vector(v)
            loglik,gradlik,_,_=self.geom(v,geom_ord=[0,1])
            Nu_r_v=self.Nu_r.T.dot(self.model.prior.M*self.v); Nu_r_v1=self.Nu_r.T.dot(self.model.prior.M*v)
            l2_norm2=lambda x: x.dot(x)
            dum0=.5*l2_norm2(Nu_r_v)+.5*l2_norm2((Nu_r_v1-self.D_Ar*Nu_r_v-self.D_Gr*(self.Nu_r.T.dot(self.gradlik)))/self.D_Br)
            dum1=.5*l2_norm2(Nu_r_v1)+.5*l2_norm2((Nu_r_v-self.D_Ar*Nu_r_v1-self.D_Gr*(self.Nu_r.T.dot(gradlik)))/self.D_Br)
            log_acpt+= dum0-dum1
        else:
            print('Wrong proposal!')
            raise
        # log of acceptance probability
        log_acpt+=-self.loglik+loglik
        if v_ref is not None:
            log_acpt+= (self.model.prior.M*v_ref).dot(self.v-v)
        # accept/reject step
        if np.isfinite(log_acpt) and np.log(np.random.uniform())<min(0,log_acpt):
            self.v[:]=v; self.loglik=loglik;
            if self.prop_name is 'LI_Langevin':
                self.gradlik[:]=gradlik;
            acpt=True
        else:
            acpt=False
        
        return acpt,log_acpt
        
    def aDRinfmHMC(self,v_ref=None):
        """
        aDRinfmHMC kernel to generate a (posterior) sample.
        Input: v, [loglik, gradlik]
        Output: v1, [loglik1, gradlik1], acpt (indicator)
        """
        # Compute a candidate using either LI-prior or LI-Langevin
        # Compute the (log) acceptance probability
        
        # initialization
        v=self.v.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
        cos_=(1-self.h/4)/(1+self.h/4); sin_=rth/(1+self.h/4)
#         cos_=np.cos(rth); sin_=np.sin(rth)
        
        # sample velocity
#         _v=self.rtK(self.model.prior.sample(whiten=True))
        _v=self.rand_aux()
        
        # weighted projected gradient
        g_r=self._g_r(v,{'LI_prior':None,'LI_Langevin':self.gradlik}[self.prop_name])
        wg=self.D_r*g_r
        
        # accumulate the power of force
        _v_r=self.Nu_v(_v)
        pw = rth/2*wg.dot(_v_r)
        
        # current energy
        E_cur = -self.loglik - self.h/8*wg.dot(wg) +0.5*_v_r.dot(self.Lambda_r*_v_r) #-0.5*sum(np.log(1+self.Lambda_r))
        
        randL=np.int(np.ceil(np.random.uniform(0,self.L)))
        
        for l in xrange(randL):
            # a half step for velocity
#             _v.axpy(rth/2,self.Nu_r.dot(wg))
            _v.axpy(rth/2,self.Nuv(wg))

            # a full step rotation
            v_=v.copy();v.zero()
            v.axpy(cos_,v_)
            v.axpy(sin_,_v)
            _v_=_v.copy();_v.zero()
            _v.axpy(cos_,_v_)
            _v.axpy(-sin_,v_)

            # update geometry
            if self.prop_name is 'LI_prior':
                loglik,_,_,_=self.geom(v)
                g_r=self._g_r(v)
            elif self.prop_name is 'LI_Langevin':
                loglik,gradlik,_,_=self.geom(v,geom_ord=[0,1])
                g_r=self._g_r(v,gradlik)
            else:
                print('Wrong proposal!')
                raise
            wg=self.D_r*g_r

            # another half step for velocity
#             _v.axpy(rth/2,self.Nu_r.dot(wg))
            _v.axpy(rth/2,self.Nuv(wg))

            # accumulate the power of force
            _v_r=self.Nu_v(_v)
            if l!=randL-1: pw+=rth*wg.dot(_v_r)
            
        # accumulate the power of force
        pw += rth/2*wg.dot(_v_r)
        
        # new energy
        E_prp = -loglik - self.h/8*wg.dot(wg) +0.5*_v_r.dot(self.Lambda_r*_v_r) #-0.5*sum(np.log(1+self.Lambda_r))

        # Metropolis test
        # log of acceptance probability
        log_acpt=-E_prp+E_cur-pw
        if v_ref is not None:
            log_acpt+= (self.model.prior.M*v_ref).dot(self.v-v)
        # accept/reject step
        if np.isfinite(log_acpt) and np.log(np.random.uniform())<min(0,log_acpt):
            self.v[:]=v; self.loglik=loglik;
            if self.prop_name is 'LI_Langevin':
                self.gradlik[:]=gradlik;
            acpt=True
        else:
            acpt=False
        
        return acpt,log_acpt
    
    def LISconvergence_diagnostic(self,Nu_r,Lambda_r):
        """
        Compute LIS convergence diagnostic, Forstner distance
        d_F(I+S_r,I+S_r1):=(sum_i log^2(lambda_i(I+S_r,I+S_r1)))^(1/2),
        S_r:=Nu_r * Lambda_r * Nu_r', lambda_i(A,B) is generalized eigen-problem of (A,B)
        Input: Nu_r, Lambda_r, Nu_r1, Lambda_r1
        Output: d_F(I+S_r,I+S_r1)
        ------------------------------------------------------------
        equiv to GEP in span(Nu_r)+span(Nu_r1): (A,B)
        where A=[Nu_r';Nu_r1'](I+S_r)[Nu_r,Nu_r1]; B=[Nu_r';Nu_r1'](I+S_r1)[Nu_r,Nu_r1]
        """
        r=Lambda_r.size;r1=self.Lambda_r.size
        k=min([r+r1,Nu_r.shape[0]])-1
#         S_r=Nu_r.dot(np.diag(Lambda_r)).dot(Nu_r.T)
#         S_r[np.diag_indices_from(S_r)]+=1
#         S_r1=self.Nu_r.dot(np.diag(self.Lambda_r)).dot(self.Nu_r.T)
#         S_r1[np.diag_indices_from(S_r1)]+=1
#         w=sps.linalg.eigsh(S_r,k=k,M=S_r1,return_eigenvectors=False) # expensive
        # project operators (I+S_r) and (I+S_r1) to V=span(Nu_r)+span(Nu_r1)
# #         Nu_r1r=self.Nu_r.T.dot(Nu_r)
#         Nu_r1r=np.array([self.Nu_r.T.dot(self.model.prior.M*r_) for r_ in Nu_r.T]).T
#         A=np.vstack([np.hstack([np.diag(1+Lambda_r),(1+Lambda_r)[:,None]*Nu_r1r.T]),
#                      np.hstack([Nu_r1r*(1+Lambda_r),np.eye(r1)+(Nu_r1r*Lambda_r).dot(Nu_r1r.T)])])
#         B=np.vstack([np.hstack([np.eye(r)+(Nu_r1r.T*self.Lambda_r).dot(Nu_r1r),Nu_r1r.T*(1+self.Lambda_r)]),
#                      np.hstack([(1+self.Lambda_r)[:,None]*Nu_r1r,np.diag(1+self.Lambda_r)])])
#         try:
#             w=sps.linalg.eigsh(A,k=k-1,M=B,return_eigenvectors=False,maxiter=1000,tol=1e-10)
#         except:
# #             w=sp.linalg.eigh(A,b=B,eigvals_only=True)
#             w=sp.linalg.eig(A,b=B,left=False,right=False)
        
        # use random eigen-decomposition
        A=lambda x: x+Nu_r.dot(Lambda_r*(Nu_r.T.dot(self.model.prior.M*x)))
        B=lambda x: x+self.Nu_r.dot(self.Lambda_r*(self.Nu_r.T.dot(self.model.prior.M*x)))
        invB=lambda x: x+self.Nu_r.dot((1./(1.+self.Lambda_r)-1.)*(self.Nu_r.T.dot(self.model.prior.M*x)))
        eigs=Eigen.geigen_RA(A,B,invB,dim=self.model.prior.dim,k=k) # only need eigenvalues (correct)
        w=eigs[0]
        
        self.d_F=np.linalg.norm(np.log(w.real[w.real>0]))
    
    def _init_h(self):
        """
        find a reasonable initial step size
        """
        h=1.
        _self=self
        sampler=getattr(_self,str(_self.alg_name))
        _self.h=h;_self.L=1;_self.form_operators()
        _,logr=sampler()
        a=2.*(np.exp(logr)>0.5)-1.
        while a*logr>-a*np.log(2):
            h*=pow(2.,a)
            _self=self
            _self.h=h;_self.L=1;_self.form_operators()
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
    
    def adaptive_MCMC(self,num_samp,num_burnin,num_retry_bad=0,**kwargs):
        """
        Algorithm 4: Adaptive function space MCMC with operator-weighted proposals.
        Require: During the LIS construction, we retain local{Nu_m,Lambda_m} to store the expected GNH evaluated from m samples;
                 and the value of d_F between the most recent two updates of the expected GNH, for LIS convergence monitoring.
        Require: At step n, given the state v_n, LIS basis Nu_r, projected empirical posterior covariance Sigma_r,
                 and operators {A, B, G} induced by {Psi_r, D_r, h}, one step of the algorithm is below. 
        """
        # determine the sampler
        try:
            sampler = getattr(self, self.alg_name)
        except AtributeError:
            print(self.alg_name, 'not found!')
        else:
            print('\nRunning adaptive dimension-reduced inf-'+self.alg_name[6:]+' now...\n')
        # allocate space to store results
        import os
        samp_fname='_samp_'+self.alg_name+'_'+self.prop_name+'_dim'+str(self.dim)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")
        samp_fpath=os.path.join(os.getcwd(),'result')
        if not os.path.exists(samp_fpath):
            os.makedirs(samp_fpath)
        self.samp=df.HDF5File(df.mpi_comm_world(),os.path.join(samp_fpath,samp_fname+".h5"),"w")
        self.logLik=np.zeros(num_samp+num_burnin)
        self.acpt=0.0 # final acceptance rate
        self.LIS_dims=[] # record the changes in the dimension of global LIS
        self.dFs=[] # record the history of Forstner distances
        self.times=np.zeros(num_samp+num_burnin) # record the history of time used for each sample
        
        # number of adaptations for step size
        if self.adpt_h:
            self.h_adpt['n_adpt']=kwargs.pop('adpt_steps',num_burnin)
        
        # initialize some recording statistics
        accp=0.0 # online acceptance
        num_cons_bad=0 # number of consecutive bad proposals
        
        beginning=timeit.default_timer()
        for n in xrange(num_samp+num_burnin):

            if n==num_burnin:
                # start the timer
                tic=timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # MH-step
            while True:
                try:
                    acpt_idx,log_acpt=sampler()
                except RuntimeError as e:
                    print(e)
                    if num_retry_bad==0:
                        acpt_idx=False; log_acpt=-np.inf
                        print('Bad proposal encountered! Passing... bias introduced.')
                        break # reject bad proposal: bias introduced
                    else:
                        num_cons_bad+=1
                        if num_cons_bad<num_retry_bad:
                            print('Bad proposal encountered! Retrying...')
                            continue # retry until a valid proposal is made
                        else:
                            acpt_idx=False; log_acpt=-np.inf # reject it and keep going
                            num_cons_bad=0
                            print(str(num_retry_bad)+' consecutive bad proposals encountered! Passing...')
                            break # reject it and keep going
                else:
                    num_cons_bad=0
                    break

            accp+=acpt_idx

            # display acceptance at intervals
            if (n+1)%100==0:
                print('\nAcceptance at %d iterations: %0.2f' % (n+1,accp/100))
                accp=0.0

            # save results
            self.logLik[n]=self.loglik
            if n>=num_burnin:
                v_f=df.Function(self.model.prior.V)
                v_f.vector()[:]=self.v
                self.samp.write(v_f,'sample_{0}'.format(n-num_burnin))
                self.acpt+=acpt_idx
            
            # adaptation of LIS
            if (n+1)%self.n_lag==0 and self.update_LIS_m<self.n_max and self.d_F>=self.threshold_LIS:
                # record the current LIS basis
                Nu_r_=self.Nu_r.copy(); Lambda_r_=self.Lambda_r.copy()
                # Update LIS
                print('\nUpdating LIS...')
                self.update_LIS(**kwargs)
                print(self.Lambda_m)
                print('Local LIS has %d dimensions; and new global LIS has %d dimensions.' % (self.dim_LIS_l,self.dim_LIS_g))
                self.LIS_dims.append(self.dim_LIS_g)
                # Update the LIS convergence diagnostic
                self.LISconvergence_diagnostic(Nu_r_,Lambda_r_)
                print('Forstner distance between two consecutive LIS'' becomes %.2e.\n' % self.d_F)
                if self.d_F<self.threshold_LIS:
                    print('\nConvergence diagnostic has dropped below the threshold. Stop updating LIS.\n')
                else:
                    self.dFs.append(self.d_F)
                # Update (low-rank) COV approximation
                self.D_r=1./(1+self.Lambda_r)
                # Update the operators
                if self.alg_name is 'aDRinfmMALA':
                    self.form_operators()
            
            # record the time
            self.times[n]=timeit.default_timer()-beginning
            
            # adapt step size h if needed
            if self.adpt_h:
                if n<self.h_adpt['n_adpt']:
                    self.h_adpt=self._dual_avg(n+1,np.exp(min(0,log_acpt)))
                    self.h=self.h_adpt['h']; self.form_operators()
                    print('New step size: %.2f; \t New averaged step size: %.6f\n' %(self.h_adpt['h'],np.exp(self.h_adpt['loghn'])))
                if n==self.h_adpt['n_adpt']:
                    self.h_adpt['h']=np.exp(self.h_adpt['loghn'])
                    self.h=self.h_adpt['h']; self.form_operators()
                    print('Adaptation completed; step size freezed at:  %.6f\n' % self.h_adpt['h'])

        # stop timer
        self.samp.close()
        toc=timeit.default_timer()
        self.time=toc-tic
        self.acpt/=num_samp
        print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
              % (self.time,num_samp,self.acpt))

        # save samples to file
        self.save_samp()
    
    def save_samp(self):
        """
        Save results to file
        """
        import os,pickle
        # create folder if not existing
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.filename=self.alg_name+'_'+self.prop_name+'_dim'+str(self.dim)+'_'+ctime
        # dump data
        f=open(os.path.join(self.savepath,self.filename+'.pckl'),'wb')
        res2save=[self.h,self.L,self.alg_name,self.prop_name,self.logLik,self.acpt,self.time,self.times,
                  self.n_lag,self.n_max,self.threshold_LIS,self.update_LIS_m,self.LIS_dims,self.dFs,self.Lambda_r,self.Nu_r]
        if self.adpt_h:
            res2save.append(self.h_adpt)
        pickle.dump(res2save,f)
        f.close()