#!/usr/bin/env python
"""
Dimension-independent likelihood-informed (DILI) MCMC
by Tiangang Cui, Kody J.H. Law, and Youssef M. Marzouk
http://www.sciencedirect.com/science/article/pii/S0021999115006701
-------------------------
Created August 4, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "1.4"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; slan@caltech.edu; lanzithinking@outlook.com"

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

class DILI:
    """
    Dimension-independent likelihood-informed (DILI) MCMC by Tiangang Cui, Kody J.H. Law, and Youssef M. Marzouk
    http://www.sciencedirect.com/science/article/pii/S0021999115006701
    ------------------------------------------------------------------
    The main purpose is to speed up the mixing of dimension-independent MCMC defined on function (Hilbert) space.
    The key idea is to find likelihood-informed (low dimensional) subspace (LIS) using prior pre-conditioned Gauss-Newton approximation of Hessian (ppGNH) averaged wrt. posterior samples;
    and apply more sophisticated/expensive methods (Langevin) in LIS and explore the complement subspace (CS) with more efficient/cheap methods like pCN.
    ------------------------------------------------------------------
    After the class is instantiated with arguments, call adaptive_MCMC to collect MCMC samples which will be stored in 'result' folder.
    """
    def __init__(self,parameter_init,model,step_size,proposal='LI_Langevin',adpt_h=False,**kwargs):
        """
        Initialize DILI MCMC instance with parameter, function providing geometric information, step size and proposal method:
        v                 : whitened parameter (vector) to sample
        model             : model to provide geometric information including log-density (likelihood), its gradient and Hessian (or Fisher) etc.
        Xi_m,Theta_m      : accumulated LIS basis-- eigenvalues and eigenvectors (eigenfunctions)
        Xi_r,Theta_r      : global LIS basis
        d_F,treshhold_LIS : Forstner distance between two covariances to diagnose the convergence of LIS if it drops below treshhold_LIS
        Sigma_r           : posterior covariance projected to LIS, estimated from empirical projected samples
        emp_mean          : empirical mean of parameter
        h                 : step size(s) of MCMC
        n_lag,n_max       : interval/maximum number of updating LIS
        n_b               : interval to update the projected (low-rank) posterior covariance approximation
        proposal          : option for proposal in MCMC
        adpt_h            : indicator to adapt step size(s)
        """
        # parameter
        self.v=parameter_init
        self.dim=parameter_init.size()
        # model
        self.model=model
        
        # sampling setting
        self.h=step_size
        self.n_lag=kwargs.pop('n_lag',200)
        self.n_max=kwargs.pop('n_max',1000)
        self.n_b=kwargs.pop('n_b',50)
        self.threshold_LIS=kwargs.pop('threshold_LIS',1e-5)
        
        target_acpt=kwargs.pop('target_acpt',0.65)
        # geometry needed
        self.geom=lambda parameter,geom_ord=[0],**kwargs: model.get_geom(parameter,geom_ord=geom_ord,whitened=True,**kwargs)
        self.loglik,self.gradlik,_,self.eigs=self.geom(self.v,geom_ord=[0,1,1.5],threshold=0.01)
        
        # LIS basis
        self.update_LIS_m=0
        self.Xi_m,self.Theta_m=self.eigs # local
        self.dim_LIS_l=self.Xi_m.size
        self.Xi_r,self.Theta_r=self.eigs # global
        self.dim_LIS_g=self.Xi_r.size
        print('Initial local/global LIS has %d dimensions.' % self.dim_LIS_l)
        self.d_F=np.inf # Forstner distance to detect convergence of LIS
        # (empirical estimate of) projected posterior covariance
        self.Sigma_r=1e-2*np.eye(self.dim_LIS_g)
#         self.Sigma_r=np.zeros((self.dim_LIS_g,self.dim_LIS_g))
        # empirical mean of parameter vector
        self.rk1update_empCOV_n=0
        self.emp_mean=df.Vector(self.v)
        # initialize re-weighted basis
        self.update_COV()
        
        # operators
        self.proposal=proposal
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
        Input: Psi_r, D_r, h
        Output: D_Ar, D_Br, D_Gr, A, B, G
        """
        # on LIS
        if self.proposal is 'LI_prior':
            self.D_Ar=(2-self.h[0]*self.D_r)/(2+self.h[0]*self.D_r)
            self.D_Br=np.sqrt(1-self.D_Ar**2)
            self.D_Gr=np.zeros_like(self.D_r)
        elif self.proposal is 'LI_Langevin':
            self.D_Ar=1-self.h[0]*self.D_r
            self.D_Br=np.sqrt(2*self.h[0]*self.D_r)
            self.D_Gr=self.h[0]*self.D_r
        else:
            print('Wrong proposal!')
            raise
        # on the complement space
        a_perp=(2-self.h[1])/(2+self.h[1])
        b_perp=np.sqrt(1-a_perp**2)
        # operators
        self.A=lambda x:self.Psi_r.dot((self.D_Ar-a_perp)*self.Psi_r.T.dot(self.model.prior.M*x)) + a_perp*x
        self.B=lambda x:self.Psi_r.dot((self.D_Br-b_perp)*self.Psi_r.T.dot(self.model.prior.M*x)) + b_perp*x
        self.G=lambda x:self.Psi_r.dot(self.D_Gr*self.Psi_r.T.dot(x)) # G only applies to an assembled gradient
        
    def update_LIS(self,threshold_l=.01,threshold_s=1e-4,threshold_g=.01):
        """
        Algorithm 1: Incremental update of the expected GNH and global LIS.
        Input: Theta_m, Xi_m, v_m1
        Output: Theta_m1, Xi_m1, Theta_r, Xi_r
        """
        # count the number of calls
        self.update_LIS_m+=1
        m=self.update_LIS_m
        # Compute the local LIS basis
        _,_,_,eigs=self.geom(self.v,geom_ord=[1.5],threshold=threshold_l)
        Lambda_m1,Phi_m1=eigs
        self.dim_LIS_l=Lambda_m1.size
        # Compute the QR decomposition
#         Q,R=np.linalg.qr(np.hstack([self.Theta_m,Phi_m1]))
        Q,R=wtQR.CholQR(np.hstack([self.Theta_m,Phi_m1]),lambda x:self.model.prior.M*x) # QR decomposition in weighted space R_M
        # Compute the new eigenvalues through the eigendecomposition
        eig_aug=np.hstack([m*self.Xi_m,Lambda_m1])
        mat_aug=(R*eig_aug).dot(R.T)/(m+1)
#         k=min([eig_aug.shape[0],mat_aug.shape[0]-1])
#         Xi_m1,W=sps.linalg.eigsh(mat_aug,k=k)
        Xi_m1,W=np.linalg.eigh(mat_aug)
        dsc_ord = Xi_m1.argsort()[::-1]
        Xi_m1 = Xi_m1[dsc_ord]; W = W[:,dsc_ord]
        # Compute the new basis
        Theta_m1=Q.dot(W)
        # truncate eigenvalues for updated LIS
        idx_s=Xi_m1>=threshold_s
        self.Theta_m=Theta_m1[:,idx_s]
        self.Xi_m=Xi_m1[idx_s]
        # truncate eigenvalues for global LIS to return
        idx_g=Xi_m1>=threshold_g
        self.Theta_r=Theta_m1[:,idx_g]
        self.Xi_r=Xi_m1[idx_g]
        self.dim_LIS_g=self.Xi_r.size
#         return self.Theta_r,self.Xi_r
    
    def rk1update_empCOV(self):
        """
        Perform rank-1 update of the empirical (projected) covariance.
        ns^2(X_n1) = (n-1)s^2(X_n) + (1-1/(n+1)) * [_X_n;x]' * [1,-1;-1,1] * [_X_n;x],
        X_n1=[X_n;x], _X_n=mean(X_n), s^2(.)=sample covariance (std if 1d)
        Input: _X_n, s^2(X_n), x
        Output: s^2(X_n1)
        """
        # count the number of calls
        self.rk1update_empCOV_n+=1
        n=self.rk1update_empCOV_n
        # projected vector
#         proj_v=self.Theta_r.T.dot(self.v)
#         rk1_v=np.vstack([self.emp_mean,proj_v])
#         rk1_v=np.vstack([self.emp_mean,self.v]).dot(self.Theta_r)
        rk1_v=np.array([self.Theta_r.T.dot(self.model.prior.M*r) for r in np.vstack([self.emp_mean,self.v])])
        # rank-1 update of empirical covariance
        rk1_update=(1-1/np.float(n+1))*rk1_v.T.dot(np.array([[1,-1],[-1,1]])).dot(rk1_v)
        self.Sigma_r=((n-1)*self.Sigma_r+rk1_update)/n
        # update empirical mean
#         self.emp_mean=(n*self.emp_mean+proj_v)/(n+1)
        self.emp_mean*=np.float_(n)/(n+1)
        self.emp_mean.axpy(1.0/(n+1),self.v)
    
    def update_COV(self):
        """
        Algorithm 2: Update of the low-rank posterior covariance approximation.
        Input: Theta_r, Sigma_r
        Output: Psi_r, D_r
        """
        # Compute eigendecomposition
        self.D_r,W_r=np.linalg.eigh(self.Sigma_r)
        self.D_r=self.D_r[::-1]; W_r=W_r[:,::-1]
        if any(self.D_r<0):
#             print('Ah-Oh--')
            self.D_r[abs(self.D_r)<1e-8] = abs(self.D_r[abs(self.D_r)<1e-8])
        # Return the reweighted LIS basis and the diagonalized covariance
        self.Psi_r=self.Theta_r.dot(W_r)
    
    def project_COV(self,Theta_r):
        """
        Algorithm 3: Posterior covariance re-projection for each LIS update.
        Input: Theta_r, Sigma_r, Theta_r1
        Output: Sigma_r1
        """
#         Theta_r1r=self.Theta_r.T.dot(Theta_r)
        self.Theta_r1r=np.array([self.Theta_r.T.dot(self.model.prior.M*r) for r in Theta_r.T]).T
#         Sigma_r_=self.Sigma_r.copy()
#         Sigma_r_[np.diag_indices_from(Sigma_r_)]-=1
#         Sigma_r1=Theta_r1r.dot(Sigma_r_).dot(Theta_r1r.T)
#         Sigma_r1[np.diag_indices_from(Sigma_r1)]+=1
#         self.Sigma_r=Sigma_r1
        self.Sigma_r=self.Theta_r1r.dot(self.Sigma_r-np.eye(self.Sigma_r.shape[0])).dot(self.Theta_r1r.T) + np.eye(self.Theta_r1r.shape[0])
#         # need to re-project empirical mean estimate of (projected) parameter
#         self.emp_mean=Theta_r1r.dot(self.emp_mean)
    
    def MH_step(self,v_ref=None):
        """
        A Metropolis-Hastings step to generate a (posterior) sample.
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
        if self.proposal is 'LI_prior':
            loglik,_,_,_=self.geom(v)
        elif self.proposal is 'LI_Langevin':
            v+=self.G(self.gradlik)
            v=self.model.prior.gen_vector(v)
            loglik,gradlik,_,_=self.geom(v,geom_ord=[0,1])
            Psi_r_v=self.Psi_r.T.dot(self.model.prior.M*self.v); Psi_r_v1=self.Psi_r.T.dot(self.model.prior.M*v)
            l2_norm2=lambda x: x.dot(x)
            dum0=.5*l2_norm2(Psi_r_v)+.5*l2_norm2((Psi_r_v1-self.D_Ar*Psi_r_v-self.D_Gr*(self.Psi_r.T.dot(self.gradlik)))/self.D_Br)
            dum1=.5*l2_norm2(Psi_r_v1)+.5*l2_norm2((Psi_r_v-self.D_Ar*Psi_r_v1-self.D_Gr*(self.Psi_r.T.dot(gradlik)))/self.D_Br)
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
            if self.proposal is 'LI_Langevin':
                self.gradlik[:]=gradlik;
            acpt=True
        else:
            acpt=False
        
        return acpt,log_acpt
    
    def LISconvergence_diagnostic(self,Theta_r,Xi_r):
        """
        Compute LIS convergence diagnostic, Forstner distance
        d_F(I+S_r,I+S_r1):=(sum_i log^2(lambda_i(I+S_r,I+S_r1)))^(1/2),
        S_r:=Theta_r * Xi_r * Theta_r', lambda_i(A,B) is generalized eigen-problem of (A,B)
        Input: Theta_r, Xi_r, Theta_r1, Xi_r1
        Output: d_F(I+S_r,I+S_r1)
        ------------------------------------------------------------
        equiv to GEP in span(Theta_r)+span(Theta_r1): (A,B)
        where A=[Theta_r';Theta_r1'](I+S_r)[Theta_r,Theta_r1]; B=[Theta_r';Theta_r1'](I+S_r1)[Theta_r,Theta_r1]
        """
        r=Xi_r.size;r1=self.Xi_r.size
        k=min([r+r1,Theta_r.shape[0]])-1
#         S_r=Theta_r.dot(np.diag(Xi_r)).dot(Theta_r.T)
#         S_r[np.diag_indices_from(S_r)]+=1
#         S_r1=self.Theta_r.dot(np.diag(self.Xi_r)).dot(self.Theta_r.T)
#         S_r1[np.diag_indices_from(S_r1)]+=1
#         w=sps.linalg.eigsh(S_r,k=k,M=S_r1,return_eigenvectors=False) # expensive
        
        # project operators (I+S_r) and (I+S_r1) to V=span(Theta_r)+span(Theta_r1)
#         Theta_r1r=self.Theta_r.T.dot(Theta_r)
#         Theta_r1r=self.Theta_r1r
#         A=np.vstack([np.hstack([np.diag(1+Xi_r),(1+Xi_r)[:,None]*Theta_r1r.T]),
#                      np.hstack([Theta_r1r*(1+Xi_r),np.eye(r1)+(Theta_r1r*Xi_r).dot(Theta_r1r.T)])])
#         B=np.vstack([np.hstack([np.eye(r)+(Theta_r1r.T*self.Xi_r).dot(Theta_r1r),Theta_r1r.T*(1+self.Xi_r)]),
#                      np.hstack([(1+self.Xi_r)[:,None]*Theta_r1r,np.diag(1+self.Xi_r)])])
#         try:
#             w=sps.linalg.eigsh(A,k=k-1,M=B,return_eigenvectors=False,maxiter=1000,tol=1e-10)
#         except:
# #             w=sp.linalg.eigh(A,b=B,eigvals_only=True)
#             w=sp.linalg.eig(A,b=B,left=False,right=False)
        
        # use random eigen-decomposition
        A=lambda x: x+Theta_r.dot(Xi_r*(Theta_r.T.dot(self.model.prior.M*x)))
        B=lambda x: x+self.Theta_r.dot(self.Xi_r*(self.Theta_r.T.dot(self.model.prior.M*x)))
        invB=lambda x: x+self.Theta_r.dot((1./(1.+self.Xi_r)-1.)*(self.Theta_r.T.dot(self.model.prior.M*x)))
        eigs=Eigen.geigen_RA(A,B,invB,dim=self.model.prior.dim,k=k) # only need eigenvalues (correct)
        w=eigs[0]
        
        self.d_F=np.linalg.norm(np.log(w.real[w.real>0]))
    
    def _init_h(self):
        """
        find a reasonable initial step size
        """
        h=np.array([.5,1.])
        _self=self
        _self.h=h;_self.form_operators()
        _,logr=self.MH_step()
        a=2.*(np.exp(logr)>0.5)-1.
        while a*logr>-a*np.log(2):
            h*=pow(2.,a)
            _self=self
            _self.h=h;_self.form_operators()
            _,logr=self.MH_step()
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
        Require: During the LIS construction, we retain local{Theta_m,Xi_m} to store the expected GNH evaluated from m samples;
                 and the value of d_F between the most recent two updates of the expected GNH, for LIS convergence monitoring.
        Require: At step n, given the state v_n, LIS basis Theta_r, projected empirical posterior covariance Sigma_r,
                 and operators {A, B, G} induced by {Psi_r, D_r, h}, one step of the algorithm is below. 
        """
        print('\nRunning adaptive DILI MCMC now...\n')
        # allocate space to store results
        import os
        samp_fname='_samp_DILI_'+self.proposal+'_dim'+str(self.dim)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")
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
                    acpt_idx,log_acpt=self.MH_step()
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
            update=False
            if (n+1)%self.n_lag==0 and self.update_LIS_m<self.n_max and self.d_F>=self.threshold_LIS:
                # record the current LIS basis
                Theta_r_=self.Theta_r.copy(); Xi_r_=self.Xi_r.copy()
                # Update LIS
                print('\nUpdating LIS...')
                self.update_LIS(**kwargs)
                print(self.Xi_m)
                print('Local LIS has %d dimensions; and new global LIS has %d dimensions.' % (self.dim_LIS_l,self.dim_LIS_g))
                self.LIS_dims.append(self.dim_LIS_g)
                # Project (posterior) COV
                self.project_COV(Theta_r_)
                # Update the LIS convergence diagnostic
                self.LISconvergence_diagnostic(Theta_r_,Xi_r_)
                print('Forstner distance between two consecutive LIS'' becomes %.2e.\n' % self.d_F)
                update=True
                if self.d_F<self.threshold_LIS:
                    print('\nConvergence diagnostic has dropped below the threshold. Stop updating LIS.\n')
                else:
                    self.dFs.append(self.d_F)
            else:
                # Perform rank-1 update of the empirical covariance
                self.rk1update_empCOV()
                if (n+1)%self.n_b==0:
                    update=True
            if update:
                # Update (low-rank) COV approximation
                self.update_COV()
                # Update the operators
                self.form_operators()
                print('Low-rank posterior covariance approximation updated!')
            
            # record the time
            self.times[n]=timeit.default_timer()-beginning
            
            # adapt step size h if needed
            if self.adpt_h:
                if n<self.h_adpt['n_adpt']:
                    self.h_adpt=self._dual_avg(n+1,np.exp(min(0,log_acpt)))
                    self.h=self.h_adpt['h']; self.form_operators()
                    print('New step size: {}; \t New averaged step size: {}\n'.format(self.h_adpt['h'],np.exp(self.h_adpt['loghn'])))
                if n==self.h_adpt['n_adpt']:
                    self.h_adpt['h']=np.exp(self.h_adpt['loghn'])
                    self.h=self.h_adpt['h']; self.form_operators()
                    print('Adaptation completed; step size frozen at:  {}\n'.format(self.h_adpt['h']))

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
        self.filename='DILI_'+self.proposal+'_dim'+str(self.dim)+'_'+ctime
        # dump data
        f=open(os.path.join(self.savepath,self.filename+'.pckl'),'wb')
        res2save=[self.h,self.proposal,self.logLik,self.acpt,self.time,self.times,
                  self.n_lag,self.n_max,self.n_b,self.threshold_LIS,self.update_LIS_m,self.LIS_dims,self.dFs,self.Xi_r,self.Theta_r]
        if self.adpt_h:
            res2save.append(self.h_adpt)
        pickle.dump(res2save,f)
        f.close()