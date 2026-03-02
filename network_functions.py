# =============================================================================
# Module: Neural Network Architecture Definitions
# Reference: Gupta et al. (2025)
# 
# Purpose:
# Defines a family of 1D Convolutional Neural Network (CNN) architectures 
# designed to map time-series decoherence data C(t) to frequency-domain 
# noise spectra S(w).
# 
# GENERIC ARCHITECTURE FLOW:
# Input: Coherence Curve Vector C(t) [Shape: (xtrain_size, 1)]
#           |
#           v
# +-----------------------+
# |  ENCODER (Conv1D)     |  -- Extracts multi-scale features
# +-----------|-----------+
#             |
#           v (MaxPooling)  -- Downsamples to isolate dominant noise features.
#             |
# +-----------|-----------+
# |  DECODER (Conv1D)     | -- Reconstructs spectral resolution.
# +-----------|-----------+
#             |
#           v (UpSampling)  -- Restores temporal/spectral dimensionality
#             |
# +-----------|-----------+
# |  OUTPUT HEAD (Dense)  | -- Maps to fixed-length Spectrum Vector S(w)
# +-----------------------+
# 
# Physical Context:
# The architectures utilize 1D convolutions to identify characteristic 
# patterns in decay curves, such as Gaussian or exponential signatures, 
# which correlate to specific noise types (1/f, white noise, or Lorentzian).
# =============================================================================

from tensorflow.keras import models, layers

def get_model( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size, net_type ):
    """
    Dispatches a specific CNN model based on the 'net_type' identifier.
    
    Inputs:
      - filter_nb (int): Base number of filters for the first convolutional layer.
      - kernel_size (int): Size of the 1D sliding window for convolutions.
      - pool_size (int): Factor used for downsampling (MaxPooling) and upsampling.
      - dropout_rate (float): Probability of setting units to 0 during training to prevent overfitting.
      - xtrain_size (int): Length of the input coherence time-series.
      - net_type (int): Integer key to select the specific network topology.
    
    Outputs:
      - model (tf.keras.Model): A compiled or uncompiled Keras Sequential model.
    """

	if net_type==1:
		model = get_model_1( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==11:
		model = get_model_11( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==2:
		model = get_model_2( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==21:
		model = get_model_21( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==3:
		model = get_model_3( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==31:
		model = get_model_31( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==4:
		model = get_model_4( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==5:
		model = get_model_5( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==6:
		model = get_model_6( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==7:
		model = get_model_7( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==8:
		model = get_model_8( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==9:
		model = get_model_9( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	
	return model


def get_model_1( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 1
    A symmetric Encoder-Decoder structure with 3 levels of down/up-sampling.
    
    Structural Logic:
      - Encoder: 2x Conv1D -> Pool -> Conv1D -> Pool -> Conv1D -> Pool.
      - Decoder: Conv1D -> UpSample -> Conv1D -> UpSample -> Conv1D -> UpSample.
      - Linear Head: Conv1D(1) -> Flatten -> Dropout -> Dense(501).
    
    Physical Context:
    This depth is designed to capture complex, non-exponential decay features
    by progressively reducing the temporal resolution and then reconstructing
    the spectral density.
    
    Inputs:
      - Same as dispatcher (filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size).
    
    Outputs:
      - model (tf.keras.Sequential): 1D CNN with a 501-node output layer.
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_11( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 11
    Similar to net_type=1 but with a lighter encoder (one Conv1D before pooling).
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_2( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 2
    Variant with a different encoder channel schedule and symmetric decoder.
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_21( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 21
    Like net_type=2 but with fewer early Conv1D layers before downsampling.
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_3( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 3
    Deeper encoder-decoder with an extra pooling/upsampling level (4 stages).
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_31( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 31
    Deepest of the 3x family: adds an additional downsampling block before decoding.
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_4( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 4
    Similar depth to net_type=3 but with different filter scaling in the encoder.
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_5( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 5
    Aggressive expansion in the decoder (filters multiplied in later layers).
    """
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb*2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb*4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb*8,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_6( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 6
    Utilizes progressively smaller kernels (kernel_size // 2, // 4, etc.)
    deeper in the network.
    
    Physical Context:
    By shrinking the kernel size at deeper layers, the model attempts to
    capture noise features at multiple frequency scales, from broadband
    white noise to narrow high-frequency resonances.
    
    Inputs:
      - filter_nb (int): Initial filter count.
      - kernel_size (int): Starting kernel width.
    
    Outputs:
      - model (tf.keras.Sequential): An encoder-heavy model with multi-scale kernels.
    """
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size//2,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//8,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//16,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D( filter_nb,kernel_size//16,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_7( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 7
    Encoder-only style (no explicit upsampling): stacked Conv/Pool followed by head.
    """
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D( filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_8( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 8
    Wider/deeper variant using an explicit per-layer filter schedule (fil_arr).
    """
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_9( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 9
    Another fil_arr-based architecture with a distinct depth/width schedule.
    """
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size//2,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//2,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size//2,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size//2,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_100( fil_arr, kernel_size, pool_size, dropout_rate, xtrain_size ):
    """
    net_type = 100
    Highly parameterized fil_arr-based encoder-decoder variant.
    """

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(fil_arr[0],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(fil_arr[1],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(fil_arr[2], kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(fil_arr[3],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( fil_arr[4],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( si5e=pool_size ) )
	model.add( layers.Conv1D( fil_arr[6],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( fil_arr[7],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( fil_arr[8],kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model
