# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:42:22 2021

@author: yokoyama
"""

from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
# get_ipython().magic('cls')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)


import sys 

fig_save_dir = current_path + '/figures/'

simName      = 'sim1'

if os.path.exists(fig_save_dir)==False:
    os.makedirs(fig_save_dir)

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 22 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 1.0
plt.rcParams['figure.dpi']       = 96
plt.rcParams['savefig.dpi']      = 600 

#%%
from sklearn import manifold
from my_modules.ar_kalman_connectivity import AR_Kalman
from my_modules.myfunc import convARcoeff2PDC, Hellinger_dist, inv_use_cholensky
from my_modules.myfunc import vis_heatmap, vis_manifold
from numpy.random import randn
import numpy as np
import sys
import joblib

sys.path.append(current_path)
#%%
param_path   = current_path + '/save_data/param/' + simName + '/'
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)
    
save_path   = current_path + '/save_data/kalman/' + simName + '/'
if os.path.exists(save_path)==False:  # Make the directory for figures
    os.makedirs(save_path)    
    

fig_save_dir = current_path + '/figures/'
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)   
#%%
def calc_model(x, p, uc, flimits):
    model = AR_Kalman(x, p, uc, flimits)
    model.est_kalman()
    
    return model, p

# def calc_KLdiv_for_parallel(mu_ref, S_ref, mu, S, idx):
#     mu1   = mu_ref
#     S1    = S_ref
#     mu2   = mu[idx,:]
#     S2    = S[:,:,idx]
    
#     kldiv = KL_div(mu1, mu2, S1, S2)
    
#     return kldiv, idx

# def calc_JSdiv_for_parallel(mu_ref, S_ref, mu, S, idx):
#     mu1   = mu_ref
#     S1    = S_ref
#     mu2   = mu[idx,:]
#     S2    = S[:,:,idx]
    
#     jsdiv = JS_div(mu1, mu2, S1, S2)
    
#     return jsdiv, idx

def calc_Hellinger_for_parallel(mu_ref, S_ref, mu, S, idx):
    mu1    = mu_ref
    S1     = S_ref
    mu2    = mu[idx,:]
    S2     = S[:,:,idx]
    
    H_dist = Hellinger_dist(mu1, mu2, S1, S2)
    
    return H_dist, idx

def calc_graph_laplacian(Adj, n_comp=2):
    from scipy.linalg import fractional_matrix_power
    A    = Adj
    D    = np.diag(np.sum(Adj, axis=1))
    dd   = np.diag(D)
    L    = D - A
    
    Lnrm = inv_use_cholensky(D) @ L
    E, V = np.linalg.eig(Lnrm)
    idx  = np.argsort(E)
    
    V    = V/dd
    V    = V[:,idx] 
    V    = np.real(V)
    Y    = V[:, 1:n_comp+1]
    # The first smallest eigenvector should be ignored, because it would have zero eigenvalue.
    # The second smallest eigenvalue is first component
    return Y

def calc_MDS(Adj, n_comp=2):
    Ndim = Adj.shape[0]
    one  = np.eye(Ndim) - np.ones((Ndim, Ndim))/Ndim
    
    # Young-Householder transformation
    # (centeralization for squared distance matrix)
    P    = - 1/2 * ( one * Adj * one)
    
    E, V = np.linalg.eig(P)
    idx  = np.flipud(np.argsort(E))
    V    = V[:, idx]
    
    Y    = V[:, :n_comp] 
    return Y
#%%
if __name__ == "__main__":
    Nch     = 10
    P       = 4  # lag order
    fs      = 1000
    Samples = 3000
    State   = 3 # Num of segment
    t       = np.linspace(0,Samples/fs-1/fs, Samples)
    flimits = np.array([1, 40])
    x       = np.zeros((Samples, Nch));
    
    coeff1        = 0.4 * np.eye(Nch)
    coeff1[1,0]   = 0.4
    coeff1[0,9]   = 0.4
    coeff1[4,7]   = 0.4
    coeff1[6,1]   = 0.4
    coeff1[3,9]   = 0.4
    A1            = np.concatenate((np.zeros((Nch,Nch,1)), 
                                    np.zeros((Nch,Nch,1)), 
                                    coeff1[:,:,np.newaxis],
                                    coeff1[:,:,np.newaxis]), axis=2)
    
    coeff2        = 0.4 * np.eye(Nch)
    coeff2[3,5]   = 0.4
    coeff2[1,3]   = 0.4
    coeff2[1,4]   = 0.4
    coeff2[0,1]   = 0.4
    coeff2[0,2]   = 0.4
    A2            = np.concatenate((np.zeros((Nch,Nch,1)), 
                                    np.zeros((Nch,Nch,1)), 
                                    coeff2[:,:,np.newaxis], 
                                    coeff2[:,:,np.newaxis]), axis=2)

    coeff3        = 0.4 * np.eye(Nch)
    coeff3[9,6:]  = 0.4
    coeff3[6,7]   = 0.4
    coeff3[4,6:]  = 0.4
    A3            = np.concatenate((np.zeros((Nch,Nch,1)), 
                                    np.zeros((Nch,Nch,1)), 
                                    coeff3[:,:,np.newaxis],
                                    coeff3[:,:,np.newaxis]), axis=2)
    
    A             = np.concatenate((A1[:,:,:,np.newaxis], A2[:,:,:,np.newaxis], A3[:,:,:,np.newaxis]), axis=3)
    
    np.random.seed(seed=0)     
    x[0:P,:] = randn(P,Nch);
    for i in range(P, Samples):
        
        if  0 <= i < 1000:
            Aij = A1
        elif 1000 <= i < 2000:
            Aij = A2
        elif i >= 2000:
            Aij = A3
        
        x[i,:] = randn(Nch)
        for p in range(1, P+1):
            x[i,:] +=  np.dot(Aij[:,:,p-1], x[i-p,:]);
    
    PDC_true = np.zeros((Nch, Nch, State))
    for sgmnt in range(3):
        PDC_true[:,:,sgmnt] = convARcoeff2PDC(A[:,:,:,sgmnt], flimits)
    #%% plot generated synthetic data
    vmin  = 0
    vmax  = 1
    
    fig   = plt.figure(figsize=(15, 18))
    outer = gridspec.GridSpec(2, 1, wspace=0.25, hspace=0.3, height_ratios=[1,0.4])
    
    inner = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=outer[0], wspace=0.2, hspace=0.5, width_ratios=[0.1,1,1,1,0.08])
    tmp   = plt.Subplot(fig, inner[:,State+1])
    ax_cb = fig.add_subplot(tmp)
    cbar_info = [False, {"orientation":"vertical", 'label': 'Coupling strength (a.u.)'}, ax_cb]
    for state in range(State):
        
        ax = plt.Subplot(fig, inner[0,state+1])
        vis_heatmap(A[:,:,2,state], vmin, vmax, ax, np.array(['Segment %d\n $A_{ij}^{(3)}$'%(state+1), 'ch $j$', 'ch $i$']), cbar_info, linewidths = 0.001, fontsize=28)
        fig.add_subplot(ax)
        if state == 0:
            ax_pos = ax.get_position()
            fig.text(ax_pos.x1 - .3, ax_pos.y1+0.03, 'A', fontsize=40)
            fig.text(ax_pos.x1 - .3, ax_pos.y1-0.5, 'B', fontsize=40)
        
        ax = plt.Subplot(fig, inner[1,state+1])
        vis_heatmap(A[:,:,3,state], vmin, vmax, ax, np.array(['\n $A_{ij}^{(4)}$', 'ch $j$', 'ch $i$']), cbar_info, linewidths = 0.001, fontsize=28)
        fig.add_subplot(ax)
        
        
        ax = plt.Subplot(fig, inner[2,state+1])
        if state == State-1:
            cbar_info = [True, {"orientation":"vertical", 'label': 'Coupling strength (a.u.)'}, ax_cb]
        elif state == 0:
            ax_pos = ax.get_position()
        vis_heatmap(PDC_true[:,:,state], vmin, vmax, ax, np.array(['\n $PDC_{ij}$', 'ch $j$', 'ch $i$']), cbar_info, linewidths = 0.001, fontsize=28)
        fig.add_subplot(ax)
    
    ax = plt.Subplot(fig, outer[1])
    ax.plot(t, x)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_xticks(np.arange(0, 4, 1))  # plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # 
    
    ylims = np.array(ax.get_ylim())
    ax.plot([1, 1], ylims, 'm--', linewidth=4, alpha=0.6)
    ax.plot([2, 2], ylims, 'm--', linewidth=4, alpha=0.6)
    ax.text(0.2, ylims[1]+20, 'Segment 1')
    ax.text(1.2, ylims[1]+20, 'Segment 2')
    ax.text(2.2, ylims[1]+20, 'Segment 3')
    fig.add_subplot(ax)
    plt.grid()
    plt.ylim(ylims)
    plt.xlim(0, 3)
    plt.savefig(fig_save_dir + 'param_setting.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + 'param_setting.svg', bbox_inches="tight")
    
    plt.show()
    #%% save synthetic data and its parameter settings
    param_dict             = {}
    param_dict['x']        = x
    param_dict['Nch']      = Nch
    param_dict['P']        = P
    param_dict['fs']       = fs
    param_dict['Nt']       = Samples
    param_dict['Nsgmnt']   = 3
    param_dict['A_true']   = A
    param_dict['PDC_true'] = PDC_true
    param_dict['t']        = t
    
    param_name             = 'param_' + simName + '.npy'
    fullpath_save_param    = param_path + param_name 
    np.save(fullpath_save_param, param_dict)
    
    del fullpath_save_param
    del param_dict
    #%%
    name     = []
    ext      = []
    for file in os.listdir(save_path):
        split_str = os.path.splitext(file)
        name.append(split_str[0])
        ext.append(split_str[1])
        
        print(split_str)
    
    #%%
    if (len(name) == 0) & (len(ext) == 0):
        P_candi = np.arange(1, 11, 1)
        UC      =  np.array([10**-i for i in range(1,8)])
        
        criteria = np.zeros((len(UC), len(P_candi)))
        
        flimits  = np.array([1, 40])
        
        for i, uc in zip(np.arange(len(UC)), UC):
            #%% Calculate time-variant AR coefficients for each model order P
            processed  = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(calc_model)(x, p, uc, flimits) for p in P_candi)
            processed.sort(key=lambda x: x[1]) # sort the output list according to the model order
            tmp_result = [tmp[0] for tmp in processed]
            #%% Determine the optimal model order
            for j in range(len(P_candi)):
                tmp          = tmp_result[j]
                k            = tmp.Kb0.shape[0]
                n            = tmp.y_hat.shape[0]
                loglike      = tmp.loglike.sum()
                
                AIC          = -2 * loglike + 2 * k
                BIC          = -2 * loglike + k * np.log(n)
                
                criteria[i,j] = np.min([AIC, BIC])
            print(uc)
        #%%
        c_min = criteria.reshape(-1).min()
        for i, uc in zip(np.arange(len(UC)), UC):
            for j, p in zip(np.arange(len(P_candi)), P_candi):
                if criteria[i,j]==c_min:
                    UC_opt = uc
                    P_opt  = p
                    break
        #%%
        est_result, _ = calc_model(x, P_opt, UC_opt, flimits)
        #%%
        save_dict                = {}
        save_dict['est_result']  = est_result
        save_dict['P_candi']     = P_candi
        save_dict['UC_candi']    = UC
        save_dict['P_opt']       = P_opt
        save_dict['UC_opt']      = UC_opt
        save_dict['criteria']    = criteria
        save_name                = 'est_result_' + simName + '.npy'
        fullpath_save            = save_path + save_name 
        np.save(fullpath_save, save_dict)
        #%%
    else:
        fullpath    = save_path + name[0] + ext[0]
        save_dict   = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
     
        est_result  = save_dict['est_result']
        P_candi     = save_dict['P_candi']
        UC          = save_dict['UC_candi']
        P_opt       = save_dict['P_opt']
        UC_opt      = save_dict['UC_opt']
        criteria    = save_dict['criteria']
        
    del save_dict
    #%%
    fig = plt.imshow(criteria, 
                     extent=[min(P_candi),max(P_candi),min(np.log10(UC)),max(np.log10(UC))],
                     )
    plt.xlabel('model order $p$')
    plt.ylabel('$\\log_{10} (UC)$')
    plt.colorbar()
    plt.show()
    #%%
    save_path               = current_path + '/save_data/features/' 
    if os.path.exists(save_path)==False:  # Make the directory for figures
        os.makedirs(save_path)
        
    name     = []
    ext      = []
    for file in os.listdir(save_path):
        split_str = os.path.splitext(file)
        name.append(split_str[0])
        ext.append(split_str[1])
        
        print(split_str)
    #%%
    t_plt = t[P_opt:]
    idx1  = np.where((t_plt >= 0) & (t_plt<1))[0]
    idx2  = np.where((t_plt >= 1) & (t_plt<2))[0]
    idx3  = np.where((t_plt >= 2) & (t_plt<3))[0]
    if (len(name) == 0) & (len(ext) == 0):
        #%% calculate the segment averaged partial directed coherence
        PDC_all    = est_result.PDC
        PDC_sgmnt1 = np.median(PDC_all[:,:,idx1], axis=2)
        PDC_sgmnt2 = np.median(PDC_all[:,:,idx2], axis=2)
        PDC_sgmnt3 = np.median(PDC_all[:,:,idx3], axis=2)
        #%% calculate sample-by-sample similarity using Hellinger divergence
        x_pred     = est_result.y_hat 
        S          = est_result.S
        
        Nt_model   = x_pred.shape[0]
        Distance_matrix = np.zeros((Nt_model, Nt_model))
        
        for i in range(Nt_model):
            mu1 = x_pred[i,:]
            S1  = S[:,:,i]
            processed  = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(calc_Hellinger_for_parallel)(mu1, S1, x_pred, S, j) for j in range(Nt_model))
            processed.sort(key=lambda x: x[1]) # sort the output list according to the order of time stamp
            dist = [tmp[0] for tmp in processed]
            
            Distance_matrix[i,:] = dist       
        #%%
        save_dict = {}
        save_dict['Distance_matrix'] = Distance_matrix
        save_dict['PDC_sgmnt1'] = PDC_sgmnt1
        save_dict['PDC_sgmnt2'] = PDC_sgmnt2
        save_dict['PDC_sgmnt3'] = PDC_sgmnt3
        
        fullpath_save = save_path + 'time_features_' + simName + '.npy'
        np.save(fullpath_save, save_dict)
        
        #%%
    else:
        #%%
        fullpath    = save_path + name[0] + ext[0]
        save_dict   = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
        
        Distance_matrix = save_dict['Distance_matrix']
        PDC_sgmnt1 = save_dict['PDC_sgmnt1']
        PDC_sgmnt2 = save_dict['PDC_sgmnt2']
        PDC_sgmnt3 = save_dict['PDC_sgmnt3']
    #%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    del save_dict
    ################## manifold learning
    Y_MDS    = calc_MDS(abs(Distance_matrix), n_comp=3)
    Y_isomap = manifold.Isomap(n_components=3, metric = 'precomputed').fit_transform(abs(Distance_matrix))
    Y_tSNE   = manifold.TSNE(n_components=3, metric = 'precomputed').fit_transform(abs(Distance_matrix))
    Y_lap    = calc_graph_laplacian(abs(Distance_matrix), n_comp=3)
    #%%
    list_sgmnt_idx = []
    list_sgmnt_idx.append(idx1)
    list_sgmnt_idx.append(idx2)
    list_sgmnt_idx.append(idx3)
    ################## visualize the result of manifold learning
    #### MDS
    fig, ax1 = vis_manifold(Y_MDS,    list_sgmnt_idx, 'MSD')
    plt.savefig(fig_save_dir + '/Hellinger/' + 'MDS.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + '/Hellinger/' + 'MDS.svg', bbox_inches="tight")
    plt.show()
    #### Isomap
    fig, ax1 = vis_manifold(Y_isomap, list_sgmnt_idx, 'Isomap')
    plt.savefig(fig_save_dir + '/Hellinger/' + 'Isomap.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + '/Hellinger/' + 'Isomap.svg', bbox_inches="tight")
    plt.show()
    #### tSNE
    fig, ax1 = vis_manifold(Y_tSNE,   list_sgmnt_idx, 'tSNE')
    plt.savefig(fig_save_dir + '/Hellinger/' + 'tSNE.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + '/Hellinger/' + 'tSNE.svg', bbox_inches="tight")
    plt.show()
    #### Graph laplacian emdedding    
    fig, ax1 = vis_manifold(Y_lap,    list_sgmnt_idx, 'Laplacian')
    plt.savefig(fig_save_dir + '/Hellinger/' + 'Laplacian.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + '/Hellinger/' + 'Laplacian.svg', bbox_inches="tight")
    plt.show()
    ####################################################################################
    #%% Visualize the estimation result of effective connectivity analysis
    fig = plt.figure(figsize=(9, 6))
    outer = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.0, width_ratios=[1,0.04])
    
    tmp   = plt.Subplot(fig, outer[:,1])
    ax_cb = fig.add_subplot(tmp)
    
    inner1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 0], wspace=0.9)
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 0], wspace=0.9)
    
    PDC_est = []
    PDC_est.append(PDC_sgmnt1)
    PDC_est.append(PDC_sgmnt2)
    PDC_est.append(PDC_sgmnt3)
    
    for i in range(State):
        a_ax = plt.Subplot(fig, inner1[i])
        vis_heatmap(PDC_true[:,:,i], vmin, vmax, a_ax, np.array(['Segment %d\n Exact'%(i+1), 'ch $j$', 'ch $i$']), cbar_info, linewidths = 0.0, fontsize=18)
        fig.add_subplot(a_ax)
        
        if i==2:
            cbar_info = [True, {"orientation":"vertical", 'label': 'PDC (a.u.)'},  ax_cb]
        else:
            cbar_info = [False, {"orientation":"vertical", 'label': 'PDC (a.u.)'},  ax_cb]
            
        b_ax = plt.Subplot(fig, inner2[i])
        vis_heatmap(PDC_est[i], vmin, vmax, b_ax, np.array(['\n Estimated', 'ch $j$', 'ch $i$']), cbar_info, linewidths = 0.0, fontsize=18)
        fig.add_subplot(b_ax)
        
    plt.savefig(fig_save_dir + 'PDC_result.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + 'PDC_result.svg', bbox_inches="tight")
    plt.show()   
    ####################################################################################