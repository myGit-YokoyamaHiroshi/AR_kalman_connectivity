# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
import numpy as np
from scipy.special import gamma, digamma, gammaln

class AR_Kalman:
    def __init__(self, x, P, Qscale, flimits):
        self.x       = x
        self.P       = P
        self.Qscale  = Qscale
        self.flimits = flimits
        
        ## parameters for gamma prior 
        self.a0    = 1
        self.b0    = 0.5
        
        self.a     = self.a0 + x.shape[0]/2
        self.b     = self.b0

##############################################################################
    def est_kalman(self):
        x           = self.x
        P           = self.P
        Qscale      = self.Qscale
        flimits     = self.flimits
        
        Nt, Nch     = x.shape
        
        self.Nch    = Nch
        
        Total_Epoch = int(Nt-P)
        
        Kb0         = np.eye(Nch*(Nch*P+1))
        mu_beta0    = np.zeros(Nch*(Nch*P+1))
        S           = np.zeros((Nch, Nch, Total_Epoch))
        self.R      = np.eye(Nch)
        self.Q      = np.diag(Qscale * np.ones(Nch*(Nch*P+1)))
        
        # loglike     = np.nan * np.ones(Total_Epoch)
        ELBO        = np.nan * np.ones(Total_Epoch)
        
        beta        = np.zeros((Total_Epoch, Nch*Nch, P))
        residual    = np.zeros((Total_Epoch, Nch))
        y_hat       = np.zeros((Total_Epoch, Nch))
        #%%
        cnt      = 0
        
        for i in range(P, Nt, 1):
            #%%
            print('Epoch: (%d / %d), index: %d'%(cnt+1, Total_Epoch, i))
            #########################################################################
            self.make_features(x, i)            
            Dx      = self.make_Dx()
            self.Dx = Dx
            #########################################################################            
            #### Prediction step
            mu_beta, Kb = self.predict(mu_beta0, Kb0)
            #### Update step : Update prior distribution (update model parameter) 
            yPred, mu_beta_new, Kb_new, s_new, elbo = self.update(mu_beta, Kb)
            ##########################################################################
            y_hat[cnt,:] = deepcopy(yPred)
            
            mu_beta0     = deepcopy(mu_beta_new)
            Kb0          = deepcopy(Kb_new)
            # loglike[cnt] = L
            ELBO[cnt]    = elbo
            S[:,:,cnt]   = deepcopy(s_new)
            
            tmp_beta = mu_beta.reshape((Nch, Nch*P+1))
            # tmp_beta = mu_beta.reshape((Nch, Nch*P))
            for p in range(P):
                idx = np.arange(0, Nch, 1) + p*Nch
                beta[cnt, :, p]  = tmp_beta[:,idx].reshape((Nch*Nch))
            
            residual[cnt, :] = tmp_beta[:,-1]
            
            cnt += 1
        
        PDC = self.calc_connectivity(beta, Nch, Nt, P, flimits)
        
        self.PDC      = PDC
        self.beta     = beta
        self.residual = residual
        # self.loglike  = loglike
        self.ELBO     = ELBO
        self.y_hat    = y_hat
        self.Kb0      = Kb0
        self.S        = S
        
        # return beta, OMEGA, Changes, L, y_hat, sigma0, Kb0
##############################################################################    
    def calc_connectivity(self,beta, Nch, Nt, P, flimits):
        frqs        = np.arange(flimits[0], flimits[-1]+1, 1)
        
        Nf           = len(frqs)
        Nt, Ndim, Np = beta.shape
        
        PDC          = np.zeros((Nch, Nch, Nt))
        for t in range(Nt):
            Aij        = np.zeros((Nch, Nch, Nf), dtype=complex)
            for f in range(Nf):
                Af_tmp = np.zeros((Nch, Nch, Np), dtype=complex)
                for p in range(Np):
                    Af_tmp[:,:,p] = beta[t, :, p].reshape(Nch, Nch) * np.exp(1j * 2 * np.pi * frqs[f] * (p+1))
                
                Aij[:,:,f] = np.eye(Nch) - np.sum(Af_tmp, axis=2)
            
            f_width    = flimits[1] - flimits[0]
            
            term1      = abs(Aij)**2
            term2      = np.sum(abs(Aij)**2, axis=1)
            term2      = term2[np.newaxis,:,:]
            
            pdc        = 1/f_width * np.sum(term1/term2, axis=2)
            
            PDC[:,:,t] = pdc
        
        return PDC
##############################################################################    
    def make_features(self, x, idx):
        Nch = self.Nch
        P   = self.P
        i   = idx
        x_train = np.flipud(x[i-P:i,:]).reshape(-1)
        x_train = np.concatenate((x_train, np.ones(1)), axis=0)
        #########################################################################
        y_train = x[i,:]
        
        self.x_train = x_train
        self.y_train = y_train
##############################################################################
    
    def make_Dx(self):#(X, T, N, P):
        X    = self.x_train
        N    = self.Nch
        
        Dx   = np.kron(np.eye(N), X)
        
        return Dx
    
    def mylogdet(self, S):
        L       = np.linalg.cholesky(S)
        logdetS = 2*np.sum(np.log(np.diag(L)))
        
        return logdetS
 
    def inv_use_cholensky(self, M):
        L     = np.linalg.cholesky(M)
        L_inv = np.linalg.inv(L)
        M_inv = np.dot(L_inv.T, L_inv)
        
        return M_inv
##############################################################################
    def predict(self, mu_beta, Kb):
        Q     = self.Q
        
        mu_beta_new = mu_beta
        Kb_new      = Kb + Q
        
        return mu_beta_new, Kb_new
        
    def update(self, mu, P):#(X, Y, Dx, mu_beta, Kb, sigma0, T, N):
        Y       = self.y_train
        H       = self.Dx
        N       = self.Nch
        R       = self.R
        R_inv   = self.inv_use_cholensky(R)
        a       = self.a + N/2
        b       = self.b
        
        eta_inv = b/a
        ############# Estimate posterior distribution #############################
        yPred   = np.dot(H, mu)
        err     = (Y.reshape(-1) - yPred.reshape(-1)) 

        S_new     = (eta_inv * R) + H @ P @ H.T 
        S_new_inv = self.inv_use_cholensky(S_new)
        #### update Kalman gain
        K         = P @ H.T @ S_new_inv # Kalman Gain
        #### update mean
        mu_new    = mu + K  @ (Y.reshape(-1) - np.dot(H, mu).reshape(-1)) 
        #### update covariance
        P_new     = P - K @ S_new @ K.T 
        
        #### update gamma prior
        b         = b + 1/2 * (err.T@R_inv@err) + 1/2 * np.trace(R_inv @ (H@P@H.T))
        ###########################################################################
        
        logdetR    = self.mylogdet(R)
        logdetP    = self.mylogdet(P)
        P_inv      = self.inv_use_cholensky(P)
        Nstate     = len(mu_new)
        
        ####### elbo
        ll_state   = 1/2 * (-Nstate * np.log(2*np.pi) - logdetP - np.trace(P_inv@P))
        ll_obs     = 1/2 * (-N * np.log(2*np.pi) +  (digamma(a) - np.log(b)) - logdetR - a/b * ((err.T@R_inv@err) + np.trace(R_inv @ (H@P@H.T))))
        ll_gamma   = (self.a-1)* (digamma(a)-np.log(b))-gammaln(self.a)+self.a*np.log(self.b)-self.b*(a/b)

        Hx         = Nstate/2 * (1 + np.log(2*np.pi)) + 1/2 * logdetP
        Heta       = a - np.log(b) + gammaln(a) + (1-a) * digamma(a)
        
        elbo       = ll_state + ll_obs + ll_gamma + Hx + Heta
        self.a     = a
        self.b     = b
        
        return yPred, mu_new, P_new, S_new, elbo



    