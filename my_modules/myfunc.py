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
#%% visualization
def vis_heatmap(Mtrx, vmin, vmax, ax, strs, cbar_info, linewidths = 0, fontsize=18): # cbar_info = [True, {"orientation":"horizontal"}, ax_cb]
    import seaborn as sns
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec
    
    if vmin < 0:       
        from matplotlib.colors import LinearSegmentedColormap
        
        cm_b = plt.get_cmap('Blues', 128)
        cm_r = plt.get_cmap('Reds', 128)
        
        color_list_b = []
        color_list_r = []
        for i in range(128):
            color_list_b.append(cm_b(i))
            color_list_r.append(cm_r(i))
        
        color_list_r = np.array(color_list_r)
        color_list_b = np.flipud(np.array(color_list_b))
        
        color_list   = list(np.concatenate((color_list_b, color_list_r), axis=0))
        
        cm = LinearSegmentedColormap.from_list('custom_cmap', color_list)
            
    elif vmin>=0:
        cm = plt.get_cmap('Reds', 256)
    
    
    title_str = strs[0]
    xlab      = strs[1]
    ylab      = strs[2]
    
    if cbar_info[0] == True:
        im = sns.heatmap(Mtrx, 
                         vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor='whitesmoke',
                         cmap=cm, 
                        cbar = True, cbar_kws = cbar_info[1], 
                        ax=ax, cbar_ax = cbar_info[2]) 
    else:
        im = sns.heatmap(Mtrx, 
                         vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor='whitesmoke',
                         cmap=cm, 
                         cbar = False, 
                         ax=ax) 
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_str, fontsize=fontsize)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')   

def vis_manifold(Y, list_sgmnt_idx, title_label):
    import matplotlib.pylab as plt
    
    cmap = plt.get_cmap('brg', len(list_sgmnt_idx))
    
    fig  = plt.figure(figsize=(6, 6))
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax1  = fig.add_subplot(111, projection='3d')
    for sgmnt in range(len(list_sgmnt_idx)):
        idx       = list_sgmnt_idx[sgmnt]
        fig_label = 'segment ' + str(sgmnt+1)
        ax1.scatter(Y[idx, 0], Y[idx, 1], Y[idx, 2], c=np.array(cmap(sgmnt))[np.newaxis, 0:3], label=fig_label);
    
    
    
    ax1.set_xlabel('axis 1', labelpad=20)
    ax1.set_ylabel('axis 2', labelpad=20)
    ax1.set_zlabel('axis 3', labelpad=20)
    ax1.set_title(title_label)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=15)
    
    return fig, ax1