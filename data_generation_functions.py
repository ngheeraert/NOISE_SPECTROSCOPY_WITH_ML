# Module: Data Generation and Preprocessing Utilities
# 
# Purpose:
# This module provides the foundational data pipeline for training Convolutional
# Neural Networks (CNNs) to perform expedited noise spectroscopy on transmon qubits.
# It bridges the gap between theoretical noise models and realistic experimental
# conditions by synthetically generating, augmenting, and formatting training datasets.
# 
# Physical & Theoretical Context:
# The functions herein implement the forward model of quantum decoherence. They
# mathematically map theoretically defined noise power spectral densities, S(w),
# to observable transverse relaxation (coherence decay) curves, C(t). This is
# achieved by numerically evaluating the overlap integral between the noise
# spectrum and the filter function of a specified dynamical decoupling sequence
# (e.g., CPMG). To ensure the trained CNN remains resilient when deployed on
# actual hardware (e.g., IBM quantum processors), the pipeline actively emulates
# experimental imperfections by injecting controlled Gaussian noise.
# 
# Core Capabilities:
#   - Forward Modeling: Constructs CPMG filter functions and computes coherence
#     decay curves via numerical integration of the decoherence functional.
#   - Hardware Alignment: Interpolates highly resolved synthetic time/frequency
#     grids onto the specific grids dictated by the physical experimental hardware.
#   - Data Augmentation & Emulation: Injects i.i.d. Gaussian noise into ideal
#     curves to simulate finite shot noise and experimental variability, creating
#     robust augmented datasets.
#   - Machine Learning Prep: Provides NumPy-native utilities for reproducible
#     train/test dataset splitting and dataset extrapolation.
# 
# Reference:
# B. Gupta et al., "Expedited Noise Spectroscopy of Transmon Qubits",
# Adv. Quantum Technol. (2025), DOI: 10.1002/qute.202500109

import numpy as np 
from scipy.interpolate import interp1d 
from tqdm import trange 


def numpy_train_test_split(X, y, test_size=0.15, shuffle=True, seed=None):
    """
    Purpose: Partitions the synthetic dataset of (coherence curve, noise spectrum)
    pairs into distinct training and testing subsets. 
    
    Inputs:
      - X (numpy.ndarray): Input data array (coherence curves). First dimension indexes samples.
      - y (numpy.ndarray): Target data array (noise spectra). Must match the first dimension of X.
      - test_size (float, default=0.15): Fraction of the total samples to reserve for testing.
      - shuffle (bool, default=True): Whether to randomize the sample indices before splitting.
      - seed (int, default=None): Random seed for reproducible shuffling.
    
    Outputs:
      - X_train (numpy.ndarray): Training subset of the input data.
      - X_test (numpy.ndarray): Testing subset of the input data.
      - y_train (numpy.ndarray): Training subset of the target data.
      - y_test (numpy.ndarray): Testing subset of the target data.
      X = np.asarray(X)
      y = np.asarray(y)
    """
    
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

def interpData(x, y, xNew, axis=-1):
    """
    Purpose: Interpolates arrays (either synthetic coherence curves or noise spectra)
    from their dense numerical generation grids onto new, specified target grids.
    Physical Context: Aligns the highly resolved synthetic time/frequency grids 
    (T_in, w0) used for theoretical integration with the specific, often sparser 
    grids corresponding to the experimental hardware capabilities (e.g., the exact 
    evolution times T_train probed on IBM Osaka qubits).
    
    Inputs:
      - x (numpy.ndarray): Original 1D grid of independent variables (time or frequency).
      - y (numpy.ndarray): Original data values corresponding to the x grid.
      - xNew (numpy.ndarray): The new 1D target grid onto which the data should be interpolated.
      - axis (int, default=-1): The axis of the y array along which to perform interpolation.
    
    Outputs:
      - interpolated_data (numpy.ndarray): The y data evaluated on the xNew grid.
    """

    print("x shape:", np.shape(x))
    print("y shape:", np.shape(y))
    f = interp1d(x, y, axis=axis)
    return f(xNew)

def prepare_trainData(c_in,T_in,T_train,noiseMax=0.02):
    """
    Purpose: Applies simulated experimental noise to the synthetic coherence curves.
    Physical Context: After generating ideal C(t) curves via the filter function,
    independent and identically distributed (i.i.d.) Gaussian noise is added. This
    step explicitly emulates the cumulative impact of experimental imperfections,
    such as finite shot noise and readout errors, collectively referred to as
    "experimental noise". Training with this noise ensures the CNN
    remains resilient and avoids predicting artifactual noise spectra when given
    imperfect, real-world data.
    
    Inputs:
      - c_in (numpy.ndarray): Array of ideal synthetic coherence curves.
      - T_in (numpy.ndarray): Original generation time grid for c_in.
      - T_train (numpy.ndarray): Target experimental time grid for interpolation.
      - noiseMax (float, default=0.02): Standard deviation of the Gaussian noise added.
    
    Outputs:
      - c_train (numpy.ndarray): Interpolated coherence curves with added synthetic Gaussian noise.
    """

    print("x shape:", np.shape(T_in))
    print("y shape:", np.shape(c_in))
    c_train = interpData(T_in,c_in,T_train)

    for i in range(c_in.shape[0]):
      c_train[i,:] = c_train[i,:] + np.random.normal(0,noiseMax,size=c_train.shape[1])

    return c_train

def generate_final_data(c_data,T_in,s_data,w0,T_train,w_train):
    """
    Purpose: Constructs the final augmented dataset by generating multiple noisy 
    realizations (controlled by 'nnps') for each underlying ideal coherence curve.
    Physical Context: This data augmentation strategy makes the CNN highly robust 
    against signal-to-noise ratio fluctuations in actual experimental runs. By 
    exposing the network to varying degrees of synthetic experimental noise during 
    training, the experimenter is freed from needing to precisely replicate 
    experimental noise levels.
    
    Inputs:
      - c_data (numpy.ndarray): Baseline ideal coherence curves.
      - T_in (numpy.ndarray): Original generation time grid for the coherence curves.
      - s_data (numpy.ndarray): Baseline target noise spectra.
      - w0 (numpy.ndarray): Original generation frequency grid for the noise spectra.
      - T_train (numpy.ndarray): Target time grid for the CNN input.
      - w_train (numpy.ndarray): Target frequency grid for the CNN output.
    
    Outputs:
      - c_train_final (numpy.ndarray): Augmented dataset containing multiple noisy copies of the coherence curves.
      - s_train_final (numpy.ndarray): Augmented dataset of corresponding noise spectra, duplicated to match the coherence curve.
    """

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

def cpmgFilter(n, Tmax):
    """
    Purpose: Calculates the timing array for the pi-pulses in a standard 
    Carr-Purcell-Meiboom-Gill (CPMG) dynamical decoupling sequence.
    Physical Context: A CPMG sequence (such as CPMG-32) acts as a specialized 
    bandpass filter. By varying the number of pulses and evolution time, this 
    protocol enables the experimental probing of the complex, high-frequency 
    regions of the noise spectrum that lie far beyond the low-frequency 1/f noise 
    regime.
    
    Inputs:
      - n (int): Number of pi-pulses in the sequence.
      - Tmax (float): Total evolution time for the pulse sequence.
    
    Outputs:
      - tpi (numpy.ndarray): 1D array containing the chronological placement times of the pi-pulses.
    """
    
    tpi = np.empty([n])
    for i in range(n):
        tpi[i]= Tmax*(((i+1)-0.5)/n)

    return tpi

def getFilter(n,w0,piLength,Tmax):
    """
    Purpose: Computes the filter function F(w*t) for a given pulse sequence.
    Physical Context: The filter function analytically models how a specific quantum 
    control protocol (like DD) isolates specific frequency components of the broad 
    environmental noise spectrum affecting the qubit. This function 
    directly represents the numerator in the overlap integral used to link noise 
    to decoherence.
    
    Inputs:
      - n (int): Number of pi-pulses in the decoupling sequence.
      - w0 (numpy.ndarray): Frequency grid over which to evaluate the filter function.
      - piLength (float): Physical duration of a single pi-pulse.
      - Tmax (float): Total evolution time of the sequence.
    
    Outputs:
      - fFunc (numpy.ndarray): Evaluated filter function magnitudes across the frequency grid.
    """

    tpi = cpmgFilter(n,Tmax)
    f = 0    
    for i in range(n):
        f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f

    fFunc = (1/2)*(( np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f) )**2)/(w0**2)

    return fFunc

def getCoherence(S,w0,T0,n,piLength):
    """
    Purpose: The forward model that maps theoretically defined noise power spectral 
    densities S(w) into synthetic coherence decay curves C(t).
    Physical Context: It numerically evaluates the mathematically ill-posed overlap 
    integral to find the decoherence functional:
       chi(t) = integral [ S(w) * F(w*t) / w^2 * dw/pi ]
    The resulting transverse relaxation (dephasing) curve is then calculated 
    assuming stationary, Gaussian noise as C(t) = exp[-chi(t)].
    
    Inputs:
      - S (numpy.ndarray): Array of noise power spectral densities.
      - w0 (numpy.ndarray): Frequency grid corresponding to the spectra.
      - T0 (numpy.ndarray): Grid of evolution times to calculate coherence over.
      - n (int): Number of pi-pulses in the applied decoupling sequence.
      - piLength (float): Duration of a single pi-pulse.
    
    Outputs:
      - C_invert (numpy.ndarray): Computed synthetic coherence decay curves (shape: samples x time steps).
    """

    steps = T0.size
    C_invert = np.empty([S.shape[0],steps,])

    for i in range(steps):
        integ = getFilter(n,np.squeeze(w0),piLength,T0[i])*S/np.pi
        integ_ans = np.trapz(y=integ,x=np.squeeze(w0))
        C_invert[:,i] = np.exp(integ_ans)

    return C_invert    

def spectrum_extend(exp_predict,w_train,w_new):
    """
    Purpose: Extrapolates the neural network's predicted noise spectra onto an 
    extended, modified frequency grid (w_new). 
    Physical Context: This utility is helpful for establishing continuous baselines
    across frequency boundaries, particularly for modeling the asymptotic behavior 
    of the noise environment—such as the constant "white noise" floor at higher 
    frequencies or dominant 1/w characteristic behavior at lower frequencies.
    
    Inputs:
      - exp_predict (numpy.ndarray): Noise spectra predicted by the CNN on the training frequency grid.
      - w_train (numpy.ndarray): Original training frequency grid.
      - w_new (numpy.ndarray): The new, extended frequency grid requiring extrapolation.
    
    Outputs:
      - s_extend (numpy.ndarray): The extended noise spectra populated across the new frequency grid.
      """

    w_lowSize = np.argwhere(w_new<w_train.min()).size
    w_lowArg = np.argwhere(w_new<w_train.min())[0]
    
    s_extend = np.zeros((exp_predict.shape[0],w_new.size))
    s_extend[:,int(w_lowArg):] = np.repeat(np.mean(exp_predict[:,-4:],axis=1),
                                          int(w_lowSize)).reshape(exp_predict.shape[0],int(w_lowSize))
    for i in range(exp_predict.shape[0]):
      s_extend[i,:int(w_lowArg)] = interpData(w_train,exp_predict[i,:],w_new[:int(w_lowArg)])

    return s_extend
