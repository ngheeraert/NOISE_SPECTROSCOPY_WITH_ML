# =============================================================================
# Data generation and preprocessing utilities.
#
# This module provides helper functions for:
#   - Splitting datasets into train/test sets (NumPy-only).
#   - Interpolating coherence curves and spectra onto desired grids.
#   - Adding controlled Gaussian noise to emulate experimental variability.
#   - Building simple CPMG-like filter functions and forward models that map
#     spectra -> coherence curves via numerical integration.
# Reference: B. Gupta et al., "Expedited Noise Spectroscopy of Transmon Qubits", Adv. Quantum Technol. (2025), DOI: 10.1002/qute.202500109
# =============================================================================

import numpy as np 
from scipy.interpolate import interp1d 
from tqdm import trange 



# -----------------------------------------------------------------------------
# NumPy-only alternative to sklearn.model_selection.train_test_split.
# - X: input array-like, first dimension indexes samples
# - y: target array-like, first dimension indexes samples (must match X)
# - test_size: fraction of samples to reserve for testing
# - shuffle/seed: reproducible random split control
# Returns: X_train, X_test, y_train, y_test
# -----------------------------------------------------------------------------
def numpy_train_test_split(X, y, test_size=0.15, shuffle=True, seed=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same first dimension, got {X.shape[0]} and {y.shape[0]}")

    n = X.shape[0]
    n_test = int(round(n * test_size))
    n_test = max(1, min(n_test, n-1))  # keep at least 1 sample in each

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# For data interpolation

# -----------------------------------------------------------------------------
# Interpolate data y(x) onto a new x-grid using scipy.interpolate.interp1d.
# The interpolation is performed along the specified axis (default: last axis).
# The function prints input shapes, which can be helpful for debugging shape
# mismatches when preparing datasets.
# -----------------------------------------------------------------------------
def interpData(x, y, xNew, axis=-1):

    print("x shape:", np.shape(x))
    print("y shape:", np.shape(y))
    f = interp1d(x, y, axis=axis)
    return f(xNew)

# For preparing training data: Add random noise, then replace low values with zeros
# Run this cell multiple times to generate sets with different random noise but same underlying curves


# -----------------------------------------------------------------------------
# Prepare a training set of coherence curves:
#   1) Interpolate each curve from T_in -> T_train
#   2) Add i.i.d. Gaussian noise (std = noiseMax) to each interpolated curve
# The added noise perturbs only the input curves (not the target spectra).
# -----------------------------------------------------------------------------
def prepare_trainData(c_in,T_in,T_train,noiseMax=0.02):
  print("x shape:", np.shape(T_in))
  print("y shape:", np.shape(c_in))
  c_train = interpData(T_in,c_in,T_train)
  for i in range(c_in.shape[0]):
    c_train[i,:] = c_train[i,:] + np.random.normal(0,noiseMax,size=c_train.shape[1])
  return c_train


# -----------------------------------------------------------------------------
# Generate a final (augmented) dataset with multiple noisy realizations.
#
# Inputs:
#   - c_data, T_in: coherence curves and their time grid (synthetic generation grid)
#   - s_data, w0: noise spectra and their frequency grid (synthetic generation grid)
#   - T_train, w_train: target grids used for training / evaluation
#
# Output:
#   - c_train_final: (nnps * N, len(T_train)) noisy coherence curves
#   - s_train_final: (nnps * N, len(w_train)) corresponding spectra (repeated)
#
# nnps controls how many noisy copies are produced per underlying sample.
# -----------------------------------------------------------------------------
def generate_final_data(c_data,T_in,s_data,w0,T_train,w_train):
    nnps = 6 #-- noise number per sample
    print("c_data shape:", np.shape(c_data))
    print("T_in shape:", np.shape(T_in))
    c_train_1set = prepare_trainData( c_data, T_in, T_train )
    s_train_1set = interpData( w0, s_data, w_train )
    d1 = np.shape( c_train_1set )[0]
    d2 = np.shape( c_train_1set )[1]
    d3 = np.shape( s_train_1set )[1]
    c_train_final = np.zeros( ( d1*nnps, d2 ) )
    s_train_final = np.zeros( ( d1*nnps, d3 ) )
    for i in range(nnps):
        c_train_1set = prepare_trainData( c_data, T_in, T_train, noiseMax=0.02 )
        c_train_final[i*d1:(i+1)*d1,:] = c_train_1set
        s_train_final[i*d1:(i+1)*d1,:] = s_train_1set

    return c_train_final, s_train_final

# %%
# Create CPMG-like pulse timing array


# -----------------------------------------------------------------------------
# Construct a CPMG-like array of π-pulse times for n pulses over [0, Tmax].
# This uses the standard midpoint placement: t_i = Tmax * ((i+1)-0.5)/n.
# -----------------------------------------------------------------------------
def cpmgFilter(n, Tmax):
    tpi = np.empty([n])
    for i in range(n):
        tpi[i]= Tmax*(((i+1)-0.5)/n)
    return tpi


# %%
# Generate filter function for a given pulse sequence


# -----------------------------------------------------------------------------
# Compute the (magnitude-squared) filter function for a pulse sequence.
#
# Parameters:
#   - n: number of π-pulses
#   - w0: frequency grid (can be a vector)
#   - piLength: π-pulse duration (used in cosine factor)
#   - Tmax: total evolution time
# Returns:
#   - fFunc: filter function evaluated on w0
# -----------------------------------------------------------------------------
def getFilter(n,w0,piLength,Tmax):
    tpi = cpmgFilter(n,Tmax)
    f = 0    
    for i in range(n):
        f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f

    fFunc = (1/2)*(( np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f) )**2)/(w0**2)
    return fFunc


# %%
# Generate decoherence curve corresponding to a noise spectrum (input shape = variable1.size x w.size)


# -----------------------------------------------------------------------------
# Forward model: compute coherence curves from spectra via numerical integration.
#
# For each evolution time in T0, the filter function is evaluated and integrated
# against S(w) to build an exponent that is then mapped to an exponential curve.
#
# Shapes (typical):
#   - S: (N_samples, N_w) or broadcastable to that
#   - w0: (N_w,)
#   - T0: (N_t,)
# Output:
#   - C_invert: (N_samples, N_t)
# -----------------------------------------------------------------------------
def getCoherence(S,w0,T0,n,piLength):
    steps = T0.size
    C_invert = np.empty([S.shape[0],steps,])
    for i in range(steps):
        integ = getFilter(n,np.squeeze(w0),piLength,T0[i])*S/np.pi
        integ_ans = np.trapz(y=integ,x=np.squeeze(w0))
        C_invert[:,i] = np.exp(integ_ans)
    return C_invert    


# -----------------------------------------------------------------------------
# Utility to extend/interpolate predicted spectra onto a new frequency grid.
#
# Given predictions on w_train, this creates an output array on w_new by:
#   - Interpolating within the overlap region
#   - Filling the remaining region with a simple constant baseline computed from
#     the tail of the prediction (mean of the last few points).
# -----------------------------------------------------------------------------
def spectrum_extend(exp_predict,w_train,w_new):
  w_lowSize = np.argwhere(w_new<w_train.min()).size
  w_lowArg = np.argwhere(w_new<w_train.min())[0]

  s_extend = np.zeros((exp_predict.shape[0],w_new.size))
  s_extend[:,int(w_lowArg):] = np.repeat(np.mean(exp_predict[:,-4:],axis=1),
                                        int(w_lowSize)).reshape(exp_predict.shape[0],int(w_lowSize))
  for i in range(exp_predict.shape[0]):
    s_extend[i,:int(w_lowArg)] = interpData(w_train,exp_predict[i,:],w_new[:int(w_lowArg)])
  return s_extend
