# AR_kalman_connectivity<br>
This is a sample of python implementation for time-varying effective connectivity analysis, based on Autoregressive (AR) model with Kalman filter. <br>

# Folder structures<br>
\AR_kalman_connectivity<br>
&ensp;&ensp;├── \figures … contains the figures generated by the script (main.py)<br>
&ensp;&ensp;│<br>
&ensp;&ensp;├─ \my_modules<br>
&ensp;&ensp;│&ensp;&ensp;&ensp;├─ ar_kalman_connectivity.py … contains main module of time-variant AR model with Kalman filter inference<br>
&ensp;&ensp;│&ensp;&ensp;&ensp;└─ myfunc.py … contains the user-defined modules<br>
&ensp;&ensp;│<br>
&ensp;&ensp;├── \save_data … contains the dataset of estimation results <br>
&ensp;&ensp;│<br>
&ensp;&ensp;└─ main.py <br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;… contains sample script for how to use of the module “ar_kalman_connectivity.py”  <br>


# Requirements<br>
&ensp; Operation has been confirmed only under the following environment. <br>
&ensp;&ensp; - OS: Windows 10 64bit, Ubuntu 18.04.5 LTS <br>
&ensp;&ensp; - Python 3.8.3, 3.8.5 <br>
&ensp;&ensp; - conda 4.8.4, 4.9.2  <br>
&ensp;&ensp; - Spyder 4.1.4, 4.1.5 <br>
&ensp;&ensp; - numpy 1.18.5, 1.19.2 <br>
&ensp;&ensp; - scipy 1.5.0, 1.5.2 <br>
&ensp;&ensp; - matplotlib 3.2.2, 3.3.2<br>
&ensp;&ensp; - networkx 2.4, 2.5 <br>
&ensp;&ensp; - IPython 7.16.1, 7.19.0 <br>
&ensp;&ensp; - joblib 0.16.0, 0.17.0 <br>
&ensp; <br>
&ensp; The implemented scripts are not guaranteed to run on any other version in Python than the above.<br>
&ensp; <br>
# Authors<br>
&ensp; Hiroshi Yokoyama<br>
&ensp;&ensp;(Division of Neural Dynamics, Department of System Neuroscience, National Institute for Physiological Sciences, Japan)<br>

# Example for use<br>
[Simple example for use]<br>
```python
# Define the module
from my_modules.ar_kalman_connectivity import AR_Kalman

# Initialize
model = AR_Kalman(x, p, uc, flimits)
### x       .... input time-series, with size of [samples x channels]
### p       .... model order for AR model
### uc      .... noise scaling factor (e.g., uc = 1E-5) 
### flimits .... frequency band to estimate the time-variant connectivity (e.g., flimits = np.array([8, 12]) ) 

# Fit the model and Estmate the time-variant connectivity
model.est_kalman()

# Get the estimation results
y_hat   = model.y_hat   # mean of posterior predictive distribution for the observation model, with size of [samples x channels]
S       = model.S       # covariance of posterior predictive distribution for the observation model, with size of [channels x channels x samples]
loglike = model.loglike # marginal log-likelihood function, with size of [samples x 1]
PDC     = model.PDC     # time-variant partial directed coherence (PDC), with size of [channels x channels x samples]
```

[Example with model selection]<br>
```python
# Define the module
from my_modules.ar_kalman_connectivity import AR_Kalman
import joblib

# self-defined function for use the "AR_Kalman" module with parapllel loop
def calc_model(x, p, uc, flimits):
    model = AR_Kalman(x, p, uc, flimits)
    model.est_kalman()
    
    return model, p

# Define candidates of the parameters
P_candi  = np.arange(1, 11, 1) # candidate of model order P
UC       =  np.array([10**-i for i in range(1,8)]) # candidate of noise scaling factor
criteria = np.zeros((len(UC), len(P_candi)))


# Determine the optimal parameter
for i, uc in zip(np.arange(len(UC)), UC):
  #%% Calculate time-variant AR coefficients for each model order P with noise scaling factor uc
  processed  = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(calc_model)(x, p, uc, flimits) for p in P_candi)
  processed.sort(key=lambda x: x[1]) # sort the output list according to the model order
  tmp_result = [tmp[0] for tmp in processed]
  #%% Determine the optimal model order
  for j in range(len(P_candi)):
    tmp          = tmp_result[j]
    k            = tmp.Kb0.shape[0]
    n            = tmp.y_hat.shape[0]
    loglike      = tmp.loglike.sum()
    
    # calculate the metric 
    AIC          = -2 * loglike + 2 * k         # Akaike information criteria
    BIC          = -2 * loglike + k * np.log(n) # Bayesian information criteria
                
    criteria[i,j] = np.min([AIC, BIC]) # select value
    
# Select the optimal parameters
# (The parameters would be selected to minimize the metric)
c_min = criteria.reshape(-1).min()
for i, uc in zip(np.arange(len(UC)), UC):
  for j, p in zip(np.arange(len(P_candi)), P_candi):
    if criteria[i,j]==c_min:
      UC_opt = uc
      P_opt  = p
      break

# Fit the model and Estmate the time-variant connectivity with optimal parameters
model, _ = calc_model(x, P_opt, UC_opt, flimits)

# Get the estimation results
y_hat   = model.y_hat   # mean of posterior predictive distribution for the observation model, with size of [samples x channels]
S       = model.S       # covariance of posterior predictive distribution for the observation model, with size of [channels x channels x samples]
loglike = model.loglike # marginal log-likelihood function, with size of [samples x 1]
PDC     = model.PDC     # time-variant partial directed coherence (PDC), with size of [channels x channels x samples]
```
