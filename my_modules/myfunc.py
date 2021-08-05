#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:03:49 2021

@author: user
"""
import numpy as np

def convARcoeff2PDC(coeff, flimits):
    Nch, _, Np = coeff.shape
    frqs       = np.arange(flimits[0], flimits[-1]+1, 1)
    Nf         = len(frqs)
    
    Aij        = np.zeros((Nch, Nch, Nf), dtype=complex)
    for f in range(Nf):
        Af_tmp = np.zeros((Nch, Nch, Np), dtype=complex)
        for p in range(Np):
            Af_tmp[:,:,p] = coeff[:, :, p] * np.exp(1j * 2 * np.pi * f * (p+1))
        
        Aij[:,:,f] = np.eye(Nch) - np.sum(Af_tmp, axis=2)
    
    f_width    = flimits[1] - flimits[0]
    
    term1      = abs(Aij)**2
    term2      = np.sum(abs(Aij)**2, axis=1)
    term2      = term2[:,np.newaxis]
    
    PDC     = 1/f_width * np.sum(term1/term2, axis=2)
    
    return PDC

def KL_div(mu1, mu2, S1, S2):
    D      = S1.shape[0]
    S1_inv = inv_use_cholensky(S1)
    S2_inv = inv_use_cholensky(S2)
    
    
    logdetS1 = mylogdet(S1)
    logdetS2 = mylogdet(S2)
    
    
    term1 = logdetS2 - logdetS1
    term2 = np.trace(S2_inv @ S1) - D
    term3 = (mu2 - mu1).T @ S2_inv @ (mu2 - mu1)
    
    KL    = 0.5 * (term1 + term2 + term3) # KL div ~  cross  -  
    return KL

def JS_div(mu1, mu2, S1, S2):
    jsdiv = (KL_div(mu1, mu2, S1, S2) + KL_div(mu2, mu1, S2, S1))/2
    
    return jsdiv

def Hellinger_dist(mu1, mu2, S1, S2):
    mu_diff  = mu1 - mu2
    S        = 0.5 * (S1 + S2)
    S_inv    = inv_use_cholensky(S)
    
    term1    = mydet(S1)**(1/4) * mydet(S2)**(1/4)
    term2    = mydet(S)**(1/2)
    exp_term = np.exp(-1/8 * mu_diff.T @ S_inv @ mu_diff)
    
    H_dist = 1 - term1/term2 * exp_term
    return H_dist

# def Wasserstein_dist(mu1, mu2, S1, S2):
#     from scipy.linalg import fractional_matrix_power
    
#     S1_root   = fractional_matrix_power(S1, 0.5)
#     quad_term =  S1_root @ S2 @ S1_root
    
#     B         = np.trace(S1 + S2 - 2 * fractional_matrix_power(quad_term, 0.5))
#     mean_diff = np.linalg.norm(mu1-mu2, ord=2)
    
#     W_dist    = mean_diff + B
    
#     return W_dist

def mylogdet(S):
    L       = np.linalg.cholesky(S)
    logdetS = 2*np.sum(np.log(np.diag(L)))
    
    return logdetS

def mydet(S):
    L       = np.linalg.cholesky(S)
    detS = np.linalg.det(L)**2
    
    return detS
      
def inv_use_cholensky(M):
    L     = np.linalg.cholesky(M)
    L_inv = np.linalg.inv(L)
    M_inv = np.dot(L_inv.T, L_inv)
    
    return M_inv