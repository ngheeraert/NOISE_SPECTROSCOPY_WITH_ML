# Module: Training and Evaluation Entry Point for Noise Spectroscopy
# Reference: Gupta et al. (2025)
# 
# Purpose:
# This script serves as the primary execution engine for training and validating
# Convolutional Neural Networks (CNNs) designed to predict noise power spectral
# densities S(w) from qubit coherence decay data C(t).
# 
# WORKFLOW DIAGRAM:
# Input: Coherence Curves C(t) from .npz or IBM Quantum Hardware
#           |
#           v
# +-----------------------+
# |  DATA PREPROCESSING   |  --> Partition into Train/Test subsets
# +-----------|-----------+
#             |
#             v
# +-----------|-----------+     +-----------------------------------------+
# |  CNN MODEL ASSEMBLY   | <-- | HYPERPARAMETERS: Filters, Kernels, etc.
# +-----------|-----------+     +-----------------------------------------+
#             |
#             v
# +-----------|-----------+
# |  TRAINING (MAPE LOSS) | --> Minimize error between predicted & true S(w) 
# +-----------|-----------+
#             |
#             v
# +-----------|-----------+     +-----------------------------------------+
# |   RESULTS & PLOTS     | --> Save .h5 model & log-log Diagnostic Plots
# +-----------------------+     +-----------------------------------------+
# 
# Physical & Theoretical Context:
# The script automates the process of mapping experimental or synthetic decoherence
# curves to their underlying noise sources. It implements a supervised
# learning workflow where the model is trained on synthetic datasets generated via
# the filter function formalism to solve the inverse problem of deconvolution.
# 
# Core Workflow:
#   - Hyperparameter Parsing: Configures network architecture and training logic via CLI.
#   - Data Ingestion: Loads pre-generated .npz datasets and partitions them for validation.
#   - Model Synthesis: Dispatches a 1D CNN architecture and compiles it using MAPE loss.
#   - Training & Optimization: Executes the fit cycle with automated learning rate reduction.
#   - Diagnostic Visualization: Generates log-log plots for performance auditing.
# 

from tensorflow.keras import models, optimizers, callbacks, Sequential
import numpy as np
import time
from network_functions import get_model
from data_generation_functions import generate_final_data, numpy_train_test_split
import sys 
import os
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['figure.dpi']=300


"""
============================================================
HYPER-PARAMETER CONFIGURATION AND CLI PARSING

Purpose:
Sets the structural and optimization parameters for the neural network. 
Values can be overridden via command-line arguments to facilitate automated 
parameter sweeps.

Inputs (Command Line Arguments):
  - --batch_size (int): Number of samples per gradient update.
  - --epochs (int): Number of complete passes through the training dataset.
  - --filters (int): Base number of convolutional filters in the initial layer.
  - --kernel_size (int): Width of the 1D convolutional window.
  - --initial_lr (float): Starting learning rate for the Adam optimizer.
  - --net_type (int): Identifier for the specific CNN architecture variant.
  - --patience (int): Number of epochs with no improvement before reducing LR.

Outputs:
  - Global variables updated to reflect user-defined or default training constraints.
 ============================================================
"""

#-- network structure
FILTERS=40
KERNEL_SIZE=5
DROPOUT_RATE=0.05
POOL_SIZE=2

#-- training hyperparamters
BATCH_SIZE=64
EPOCHS=20
INITIAL_LR=1e-3
MIN_LR=1e-6
RED_FACTOR=0.5
DROPOUT_RATE=0.05
PATIENCE=6
MIN_DELTA=0.5
NB=0
VERBOSE=True
NET_TYPE=1

n = len(sys.argv)

for i in range(1, n):
	if isinstance( sys.argv[i], str ):
		if (sys.argv[i]=='--batch_size'):
			BATCH_SIZE=int( sys.argv[i+1] )
		elif (sys.argv[i]=='--epochs'):
			EPOCHS=int(sys.argv[i+1])
		elif (sys.argv[i]=='--filters'):
			FILTERS=int(sys.argv[i+1])
		elif (sys.argv[i]=='--kernel_size'):
			KERNEL_SIZE=int(sys.argv[i+1])
		elif (sys.argv[i]=='--initial_lr'):
			INITIAL_LR=float(sys.argv[i+1])
		elif (sys.argv[i]=='--min_lr'):
			MIN_LR=float(sys.argv[i+1])
		elif (sys.argv[i]=='--min_delta'):
			MIN_DELTA=float(sys.argv[i+1])
		elif (sys.argv[i]=='--patience'):
			PATIENCE=float(sys.argv[i+1])
		elif (sys.argv[i]=='--nb'):
			NB=int(sys.argv[i+1])
		elif (sys.argv[i]=='--verbose'):
			if int(sys.argv[i+1])==1:
				VERBOSE=True
			else:
				VERBOSE=False
		elif (sys.argv[i]=='--net_type'):
			NET_TYPE=int(sys.argv[i+1])

print('=============================')
print('-- BATCH_SIZE  = ', BATCH_SIZE)
print('-- EPOCHS      = ', EPOCHS)
print('-- FILTERS     = ', FILTERS)
print('-- KERNEL_SIZE = ', KERNEL_SIZE)
print('-- NET_TYPE    = ', NET_TYPE)
print('-- INITIAL_LR  = ', INITIAL_LR)
print('-- MIN_LR      = ', MIN_LR)
print('-- MIN_DELTA   = ', MIN_DELTA)
print('-- PATIENCE    = ', PATIENCE)
print('=============================')

"""
============================================================
DATA INGESTION AND PREPROCESSING

Purpose: Loads synthetic or experimental decoherence datasets and prepares 
the input (X) and target (y) tensors for the CNN.

Physical Context: The 'c_in' data represents the noisy coherence decay C(t), 
while 's_in'  represents the target noise power spectral density S(w).

Inputs:
  - data_file_name (.npz): Compressed NumPy file containing 'c_in', 's_in', 
    and associated time/frequency grids (T_train, w_train).

Outputs:
  - x_train, x_test (numpy.ndarray): Training and validation sets of 
    decoherence curves C(t).
  - y_train, y_test (numpy.ndarray): Training and validation sets of 
    target noise spectra S(w).
============================================================
"""

data_file_name='Mar14_x32_noisy_20_noises'
data = np.load("data/"+data_file_name+".npz")

c_data = data['c_in'] 
T_in = data['T_in']    # Time vector for data generation 
s_data = data['s_in'] 
w0 = data['w_in']           # Omega vector for data generation
T_train = data['T_train']      # Time vector for training data (based on the experimental data)
w_train = data['w_train']      # Omega vector for training data
print('-- data loaded from: '+data_file_name+".npz")

x_train, x_test, y_train, y_test = numpy_train_test_split( c_data, s_data, test_size=0.15, shuffle=True, seed=2)

print("  x_train = ",np.shape(x_train))
print("  y_train = ",np.shape(y_train))
print("  x_test = ",np.shape(x_test))
print("  y_test = ",np.shape(y_test))
print("  w0 = ",np.shape(w0))
print("  w_train = ",np.shape(w_train))


"""
MODEL COMPILATION AND OPTIMIZATION LOOP

Purpose: Initializes the CNN, defines the loss function, and executes the 
backpropagation loop.

Physical Context: Uses MAPE (Mean Absolute Percentage Error) as the loss function, 
which is particularly effective for noise spectra that span several orders 
of magnitude in amplitude.

Inputs:
  - net_type (int): Selects the specific encoder-decoder depth.
  - RED_FACTOR (float): Factor by which the learning rate is multiplied 
    during a plateau.

Outputs:
  - model (tf.keras.Model): The trained neural network.
  - history_ (callback): Object containing per-epoch loss and validation metrics.
"""

X_TRAIN_SIZE = np.shape(x_train)[-1]

model = get_model( filter_nb=FILTERS, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE,\
                  dropout_rate=DROPOUT_RATE, xtrain_size=X_TRAIN_SIZE, net_type=NET_TYPE )  #-- create model

#-- prepare the model and the callback function
reduce_lr = callbacks.ReduceLROnPlateau( monitor="loss", \
										factor=RED_FACTOR,\
										patience=PATIENCE,\
										verbose=VERBOSE,\
										mode="auto",\
										min_delta=MIN_DELTA,\
										cooldown=0,\
										min_lr=MIN_LR)  #-- define LR reduction strategy

#-- define optimizer
opt = optimizers.Adam(learning_rate=INITIAL_LR)  #-- define optimizer

#-- compile
model.compile(loss='MAPE', optimizer=opt)  #-- compilation

print('-- beginning fit')
t1 = time.time()
history_ = model.fit( x_train, y_train, BATCH_SIZE, epochs=EPOCHS,\
                        validation_data=(x_test, y_test), verbose=VERBOSE, callbacks=[reduce_lr])

final_accuracy = np.round(history_.history['val_loss'][-1],2)
print('-- fit complete')
print('-- time taken=', time.time() - t1)
print('-- final accuracy=', final_accuracy)

#-- define useful paramchar
paramchar_no_fil_no_ker="dr"+str(DROPOUT_RATE)\
            +"_ps="+str(POOL_SIZE)+'_LRini='+str(INITIAL_LR)+'_LRmin='+str(MIN_LR)\
            +'_bs='+str(BATCH_SIZE)+"_ep="+str(EPOCHS)+"_nt="+str(NET_TYPE)
paramchar=str( np.round(history_.history['val_loss'][-1],2) )\
			+"_fil="+str(FILTERS)+"_ker="+str(KERNEL_SIZE)+"_dr"+str(DROPOUT_RATE)\
            +"_ps="+str(POOL_SIZE)+'_LRini='+str(INITIAL_LR)+'_LRmin='+str(MIN_LR)\
            +'_bs='+str(BATCH_SIZE)+"_ep="+str(EPOCHS)+"_nt="+str(NET_TYPE)\
            +'_'+ data_file_name
print("-- paramchar = "+paramchar)

#-- save the model
model.save('TRAINED_NETWORKS/MODEL_'+paramchar+'.h5', overwrite=True)
print("Saved:", 'TRAINED_NETWORKS/MODEL_'+paramchar+'.h5')

#filename = 'TRAINED_NETWORKS/accuracy_'+paramchar_no_fil_no_ker+'.txt'
#if os.path.exists(filename):
#    append_write = 'a' # append if already exists
#else:
#    append_write = 'w' # make a new file if not
#
#f = open( filename, append_write )
#f.write( str(FILTERS) +'    '+ str(KERNEL_SIZE) + '    ' + str( np.round(history_.history['val_loss'][-1],2) ) + '\n' )
#f.close()

"""
PERFORMANCE EVALUATION AND DIAGNOSTIC PLOTTING

Purpose:
Quantifies the model's predictive accuracy and generates visual 
benchmarks against the test set.

Physical Context:
Generates log-log plots of Noise Amplitude vs. Frequency to verify 
if the model correctly identifies characteristic noise profiles like 
1/f behavior or white noise floors.

Outputs:
  - VAL_ACC_HISTORY.pdf: Training/Validation loss curves.
  - MODEL_TEST.pdf: Comparison of true vs. predicted S(w) for 
    randomly selected test samples.
  - Saved .h5 Model: The serialized weights and architecture for later 
    deployment on experimental data.
"""

plt.subplot(1, 1, 1)
plt.axhline(y = final_accuracy, color = 'black', dashes=[2,2,2,2])
plt.title('Final accuracy = '+str(final_accuracy))
plt.plot( np.arange( 0, EPOCHS ) , history_.history['val_loss'], label='validation' )
plt.plot( np.arange( 0, EPOCHS ) , history_.history['loss'], label='training' )
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(-1,100)
plt.legend()
plt.savefig( 'TRAINED_NETWORKS/VAL_ACC_HISTORY_'+paramchar+'.pdf', format='pdf' )
plt.close()

#-- apply the network to the whole test set
predictions = model.predict(x_test)

#-- plot randomly selected noise spectra and compare
plt.subplot(1, 1, 1)
rand_set = np.random.randint( 0, y_test.shape[0] ,(8,) )
for i in rand_set:
    plt.plot( w_train/1e6, y_test[i,:], color='C'+str(i) )
    plt.plot( w_train/1e6, predictions[i], dashes=[2,2,2,2], color='C'+str(i) )
plt.xscale('log')
plt.ylim(1e2,5e4)
plt.yscale('log')
plt.ylabel('Noise Amplitude')
plt.xlabel('Frequency \omega (MHz*2*pi)')
plt.title('Final accuracy = '+str(final_accuracy))
plt.savefig( 'TRAINED_NETWORKS/MODEL_TEST_'+paramchar+'.pdf', format='pdf' )
plt.close()

