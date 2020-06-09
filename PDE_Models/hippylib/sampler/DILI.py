#!/usr/bin/env python
"""
Dimension-independent likelihood-informed (DILI) MCMC
by Tiangang Cui, Kody J.H. Law, and Youssef M. Marzouk
http://www.sciencedirect.com/science/article/pii/S0021999115006701
-------------------------
Created July 15, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; lanzithinking@outlook.com"

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import timeit,time

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
    def __init__(self,parameter_init,geometry_fun,step_size,proposal='LI_Langevin'):
        """
        Initialize DILI MCMC instance with parameter, function providing geometric information, step size and proposal method:
        v                 : parameter to sample
        geom              : function to provide geometric information including log-density (likelihood), its gradient and Hessian (or Fisher) etc.
        Xi_m,Theta_m      : local LIS basis-- eigenvalues and eigenvectors (eigenfunctions)
        Xi_r,Theta_r      : global LIS basis
        d_F,treshhold_LIS : Forstner distance between two covariances to diagnose the convergence of LIS if it drops below treshhold_LIS
        Sigma_r           : posterior covariance projected to LIS, estimated from empirical projected samples
        emp_mean          : empirical mean of parameter projected to LIS
        h                 : step size(s) of MCMC
        n_lag,n_max       : interval/maximum number of updating LIS
        n_b               : interval to update the projected (low-rank) posterior covariance approximation
        proposal          : option for proposal in MCMC
        """
        # parameter
        self.v=parameter_init
        self.dim=parameter_init.size
        
        # geometry needed
        self.geom=geometry_fun
        self.ll,self.g,self.HessApply,self.eigs=self.geom(self.v,[0,1,1.5,2],k=100)
        
        # LIS basis
        self.update_LIS_m=0
        self.Xi_m,self.Theta_m=self.eigs # local
        self.Xi_r,self.Theta_r=self.eigs # global
        self.d_F=np.inf # Forstner distance to detect convergence of LIS
        # (empirical estimate of) projected posterior covariance
        self.Sigma_r=.1*np.eye(self.Xi_r.size)
        # empirical mean of projected vector
        self.rk1update_empCOV_n=0
        self.emp_mean=self.Theta_r.T.dot(self.v)
        # initialize reweighted basis
        self.update_COV()
        
        # sampling setting
        self.h=step_size
        self.n_lag=100
        self.n_max=1000
        self.n_b=50
        self.threshold_LIS=1e-3
        
        # operators
        self.proposal=proposal
        self.form_operators()
    
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
        self.A=self.Psi_r.dot(np.diag(self.D_Ar-a_perp)).dot(self.Psi_r.T)
        self.A[np.diag_indices_from(self.A)]+=a_perp
        self.B=self.Psi_r.dot(np.diag(self.D_Br-b_perp)).dot(self.Psi_r.T)
        self.B[np.diag_indices_from(self.B)]+=b_perp
        self.G=self.Psi_r.dot(np.diag(self.D_Gr)).dot(self.Psi_r.T)
        
    def update_LIS(self,threshold_l=.1,threshold_s=1e-4,threshold_g=.1):
        """
        Algorithm 1: Incremental update of the expected GNH and global LIS.
        Input: Theta_m, Xi_m, v_m1
        Output: Theta_m1, Xi_m1, Theta_r, Xi_r
        """
        # count the number of calls
        self.update_LIS_m+=1
        m=self.update_LIS_m
        # Compute the local LIS basis
#         _,_,_,eigs=self.geom(self.v,[2],threshold_l)
        _,_,_,eigs=self.geom(self.v,[2],k=50)
        Lambda_m1,Phi_m1=eigs
        # Compute the QR decomposition
        Q,R=np.linalg.qr(np.hstack([self.Theta_m,Phi_m1]))
        # Compute the new eigenvalues through the eigendecomposition
        eig_aug=np.diag(np.hstack((m*self.Xi_m,Lambda_m1)))
        mat_aug=R.dot(eig_aug.dot(R.T))/(m+1)
        k=min([eig_aug.shape[0],mat_aug.shape[0]-1])
        Xi_m1,W=sps.linalg.eigsh(mat_aug,k=k)
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
        return self.Theta_r,self.Xi_r
    
    def rk1update_empCOV(self):
        """
        Perform rank-1 update of the empirical (projected) covariance.
        ns^2(X_n1) = (n-1)s^2(X_n) + (1-1/(n+1)) * [_X_n;x]' * [1,-1;-1,1] * [_X_0;x],
        X_n1=[X_n;x], _X_n=mean(X_n), s^2(.)=sample covariance (std if 1d)
        Input: _X_n, s^2(X_n), x
        Output: s^2(X_n1)
        """
        # count the number of calls
        self.rk1update_empCOV_n+=1
        n=self.rk1update_empCOV_n
        # projected vector
        proj_v=self.Theta_r.T.dot(self.v)
        rk1_v=np.vstack([self.emp_mean,proj_v])
        # rank-1 update of empirical covariance
        rk1_update=(1-1/np.float(n+1))*rk1_v.T.dot(np.array([[1,-1],[-1,1]])).dot(rk1_v)
        self.Sigma_r=((n-1)*self.Sigma_r+rk1_update)/n
        # update empirical mean
        self.emp_mean=(n*self.emp_mean+proj_v)/(n+1)
    
    def update_COV(self):
        """
        Algorithm 2: Update of the low-rank posterior covariance approximation.
        Input: Theta_r, Sigma_r
        Output: Psi_r, D_r
        """
        # Compute eigendecomposition
        self.D_r,W_r=np.linalg.eigh(self.Sigma_r)
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
        Theta_r1r=self.Theta_r.T.dot(Theta_r)
        Sigma_r_=self.Sigma_r.copy()
        Sigma_r_[np.diag_indices_from(Sigma_r_)]-=1
        Sigma_r1=Theta_r1r.dot(Sigma_r_).dot(Theta_r1r.T)
        Sigma_r1[np.diag_indices_from(Sigma_r1)]+=1
        self.Sigma_r=Sigma_r1
    
    def MH_step(self,v_ref=None):
        """
        A Metropolis-Hastings step to generate a (posterior) sample.
        Input: v, [ll, g]
        Output: v1, [ll1, g1], acpt (indicator)
        """
        if v_ref is None:
            v_ref=np.zeros_like(self.v)
        # Compute a candidate using either LI-prior or LI-Langevin
        # Compute the (log) acceptance probability
        v=self.A.dot(self.v)+self.B.dot(np.random.randn(self.v.size))
        if self.proposal is 'LI_prior':
            log_acpt=.5*self.v.dot(self.v)-.5*v.dot(v) + v_ref.dot(self.v-v)
        elif self.proposal is 'LI_Langevin':
            v-=self.G.dot(self.g)
            ll,g,_,_=self.geom(v,[0,1])
            tmp0=.5*np.linalg.norm(self.Psi_r.T.dot(self.v))**2+.5*np.linalg.norm((self.Psi_r.T.dot(v)-self.D_Ar*(self.Psi_r.T.dot(self.v))+self.D_Gr*(self.Psi_r.T.dot(self.g)))/self.D_Br)**2
            tmp1=.5*np.linalg.norm(self.Psi_r.T.dot(v))**2+.5*np.linalg.norm((self.Psi_r.T.dot(self.v)-self.D_Ar*(self.Psi_r.T.dot(v))+self.D_Gr*(self.Psi_r.T.dot(g)))/self.D_Br)**2
            log_acpt=self.ll-ll + v_ref.dot(self.v-v) + tmp0-tmp1
        else:
            print('Wrong proposal!')
            raise
        # accept/reject step
        if np.isfinite(log_acpt) and np.log(np.random.uniform())<min(0,log_acpt):
            self.v=v
            acpt=True
        else:
            acpt=False
        if acpt and self.proposal is 'LI_Langevin':
            self.ll=ll; self.g=g;
        return acpt
    
    def LISconvergence_diagnostic(self,Theta_r,Xi_r):
        """
        Compute LIS convergence diagnostic, Forstner distance
        d_F(I+S_r,I+S_r1):=(sum_i log^2(lambda_i(I+S_r,I+S_r1)))^(1/2),
        S_r:=Theta_r * Xi_r * Theta_r', lambda_i(A,B) is generalized eigen-problem of (A,B)
        Input: Theta_r, Xi_r, Theta_r1, Xi_r1
        Output: d_F(I+S_r,I+S_r1)
        """
        r=Xi_r.size;r1=self.Xi_r.size
        k=min([r+r1,Theta_r.shape[0]-1])
        S_r=Theta_r.dot(np.diag(Xi_r)).dot(Theta_r.T)
        S_r[np.diag_indices_from(S_r)]+=1
        S_r1=self.Theta_r.dot(np.diag(self.Xi_r)).dot(self.Theta_r.T)
        S_r1[np.diag_indices_from(S_r1)]+=1
        w=sps.linalg.eigsh(S_r,k=k,M=S_r1,return_eigenvectors=False) # expensive
#         # project operators (I+S_r) and (I+S_r1) to V=span(Theta_r)+span(Theta_r1)
#         Theta_r1r=self.Theta_r.T.dot(Theta_r)
#         I_plus_Xi_r=np.diag(1+Xi_r);I_plus_Xi_r1=np.diag(1+self.Xi_r)
#         L=np.vstack([np.hstack([I_plus_Xi_r,I_plus_Xi_r.dot(Theta_r1r.T)]),
#                      np.hstack([Theta_r1r.dot(I_plus_Xi_r),np.eye(r1)+Theta_r1r.dot(np.diag(Xi_r)).dot(Theta_r1r.T)])])
#         R=np.vstack([np.hstack([np.eye(r)+Theta_r1r.T.dot(np.diag(self.Xi_r)).dot(Theta_r1r),Theta_r1r.T.dot(I_plus_Xi_r1)]),
#                      np.hstack([I_plus_Xi_r1.dot(Theta_r1r),I_plus_Xi_r1])])
#         w=sps.linalg.eigsh(L,k=k,M=R,return_eigenvectors=False)
        self.d_F=np.linalg.norm(np.log(w[w>0]))
    
    def adaptive_MCMC(self,num_samp,num_burnin,num_retry_bad=0):
        """
        Algorithm 4: Adaptive function space MCMC with operator-weighted proposals.
        Require: During the LIS construction, we retain local{Theta_m,Xi_m} to store the expected GNH evaluated from m samples;
                 and the value of d_F between the most recent two updates of the expected GNH, for LIS convergence monitoring.
        Require: At step n, given the state v_n, LIS basis Theta_r, projected empirical posterior covariance Sigma_r,
                 and operators {A, B, G} induced by {Psi_r, D_r, h}, one step of the algorithm is below. 
        """
        print('\nRunning adaptive DILI MCMC now...\n')
        # allocate space to store results
        self.samp=np.zeros((num_samp,self.dim))
        self.loglik=np.zeros(num_samp+num_burnin)
        accp=0.0 # online acceptance
        self.acpt=0.0 # final acceptance rate
        num_cons_bad=0 # number of consecutive bad proposals
        
        for n in range(num_samp+num_burnin):

            if n==num_burnin:
                # start the timer
                tic=timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # MH-step
            while True:
                try:
                    acpt_idx=self.MH_step()
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
            if (n+1)%100==0:
                print('\nAcceptance at %d iterations: %0.2f' % (n+1,accp/100))
                accp=0.0

            # save results
            self.loglik[n]=self.ll
            if n>=num_burnin:
                self.samp[n-num_burnin,]=self.v.T
                self.acpt+=acpt_idx
            
            # adaptation of LIS
            update=False
            if (n+1)%self.n_lag==0 and self.update_LIS_m<self.n_max and self.d_F>=self.threshold_LIS:
                # record the current LIS basis
                Theta_r_=self.Theta_r.copy(); Xi_r_=self.Xi_r.copy()
                # Update LIS
                print('\nUpdating LIS...')
                self.update_LIS()
                # Project (posterior) COV
                self.project_COV(Theta_r_)
                # Update the LIS convergence diagnostic
                self.LISconvergence_diagnostic(Theta_r_,Xi_r_)
                update=True
                print('New LIS has %d dimensions.' % self.Xi_r.size)
                print('Forstner distance between two consecutive LIS becomes %.2e.\n' % self.d_F)
                if self.d_F<self.threshold_LIS:
                    print('\nConvergence diagnostic has dropped below the threshold. Stop updating LIS.\n')
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

        # stop timer
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
        import os
        import pickle
        # create folder if not existing
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.filename=self.alg_name+'_dim'+str(self.dim)+'_'+ctime
        # dump data
        f=open(os.path.join(self.savepath,self.filename+'.pckl'),'wb')
        pickle.dump([self.h,self.proposal,self.samp,self.loglik,self.acpt,self.time,
                     self.n_lag,elf.n_max,elf.n_b,self.threshold_LIS,self.update_LIS.m,self.Xi_r,self.Theta_r,self.d_F],f)
        f.close()