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

if os.name == 'posix': # for linux
    os.chdir('/home/user/Documents/Python_Scripts/sim_AR_kalman_connectivity')
elif os.name == 'nt': # for windows
    os.chdir('D:/Python_Scripts/sim_AR_kalman_connectivity')

current_path = os.getcwd()
fig_save_dir = current_path + '/figures/'

simName      = 'sim1'

if os.path.exists(fig_save_dir)==False:
    os.makedirs(fig_save_dir)

import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 1.0
plt.rcParams['figure.dpi']       = 96
plt.rcParams['savefig.dpi']      = 600 

#%%
from sklearn import manifold
from my_modules.ar_kalman_connectivity import AR_Kalman
from my_modules.myfunc import convARcoeff2PDC, JS_div, KL_div, Hellinger_dist, inv_use_cholensky
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
    
    
    sd   = P.std(axis=0)
    sd   = sd[idx]
    
    Y    = V[:, :n_comp] * sd[:n_comp]
    return Y
#%%
if __name__ == "__main__":
    Nch     = 10
    P       = 4  # lag order
    fs      = 1000
    Samples = 3000
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
    
    PDC_true = np.zeros((Nch, Nch, 3))
    for sgmnt in range(3):
        PDC_true[:,:,sgmnt] = convARcoeff2PDC(A[:,:,:,sgmnt], flimits)
    
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
        #%% calculate sample-by-sample similarity using JS divergence
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
        #%%
    del save_dict
    
    #%% MDS
    Y_MDS = calc_MDS(abs(Distance_matrix), n_comp=3)
    
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(Y_MDS[idx1, 0], Y_MDS[idx1, 1], Y_MDS[idx1, 2], c='b', label='segment 1');
    ax1.scatter(Y_MDS[idx2, 0], Y_MDS[idx2, 1], Y_MDS[idx2, 2], c='r', label='segment 2');
    ax1.scatter(Y_MDS[idx3, 0], Y_MDS[idx3, 1], Y_MDS[idx3, 2], c='g', label='segment 3');
    
    ax1.set_xlabel('axis 1')
    ax1.set_ylabel('axis 2')
    ax1.set_zlabel('axis 3')
    ax1.set_title('MDS')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=15)
    plt.show()
    #%%
    Y_isomap = manifold.Isomap(n_components=3, metric = 'precomputed').fit_transform(abs(Distance_matrix))
    
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(Y_isomap[idx1, 0], Y_isomap[idx1, 1], Y_isomap[idx1, 2], c='b', label='segment 1');
    ax1.scatter(Y_isomap[idx2, 0], Y_isomap[idx2, 1], Y_isomap[idx2, 2], c='r', label='segment 2');
    ax1.scatter(Y_isomap[idx3, 0], Y_isomap[idx3, 1], Y_isomap[idx3, 2], c='g', label='segment 3');
    
    ax1.set_xlabel('axis 1')
    ax1.set_ylabel('axis 2')
    ax1.set_zlabel('axis 3')
    ax1.set_title('Isomap')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=15)
    plt.show()
    #%%
    Y_tSNE = manifold.TSNE(n_components=3, metric = 'precomputed').fit_transform(abs(Distance_matrix))
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(Y_tSNE[idx1, 0], Y_tSNE[idx1, 1], Y_tSNE[idx1, 2], c='b', label='segment 1');
    ax1.scatter(Y_tSNE[idx2, 0], Y_tSNE[idx2, 1], Y_tSNE[idx2, 2], c='r', label='segment 2');
    ax1.scatter(Y_tSNE[idx3, 0], Y_tSNE[idx3, 1], Y_tSNE[idx3, 2], c='g', label='segment 3');
    
    
    ax1.set_xlabel('axis 1')
    ax1.set_ylabel('axis 2')
    ax1.set_zlabel('axis 3')
    ax1.set_title('tSNE')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=15)
    plt.show()
    #%%
    Y_lap  = calc_graph_laplacian(abs(Distance_matrix), n_comp=3)
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(Y_lap[idx1, 0], Y_lap[idx1, 1], Y_lap[idx1, 2], c='b', label='segment 1');
    ax1.scatter(Y_lap[idx2, 0], Y_lap[idx2, 1], Y_lap[idx2, 2], c='r', label='segment 2');
    ax1.scatter(Y_lap[idx3, 0], Y_lap[idx3, 1], Y_lap[idx3, 2], c='g', label='segment 3');
    
    
    ax1.set_xlabel('axis 1')
    ax1.set_ylabel('axis 2')
    ax1.set_zlabel('axis 3')
    ax1.set_title('Laplacian')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1), fontsize=15)
    plt.show()