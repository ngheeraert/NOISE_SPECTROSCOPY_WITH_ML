# =============================================================================
# Neural network architecture definitions.
#
# This module defines a family of 1D convolutional models (net_type) that map a
# single-channel time-series input to a fixed-length spectrum vector (Dense(501)).
#
# Common pattern across most variants:
#   - Encoder: repeated Conv1D + MaxPooling1D blocks to extract features / downsample
#   - Decoder: Conv1D + UpSampling1D blocks to upsample / reconstruct higher resolution
#   - Head: Conv1D(1) -> Flatten -> Dropout -> Dense(501)
#
# The 'get_model(...)' dispatcher selects the variant by 'net_type'.
# Reference: B. Gupta et al., "Expedited Noise Spectroscopy of Transmon Qubits", Adv. Quantum Technol. (2025), DOI: 10.1002/qute.202500109
# =============================================================================

from tensorflow.keras import models, layers

# -----------------------------------------------------------------------------
# Dispatcher: returns the model corresponding to 'net_type'.
# -----------------------------------------------------------------------------
def get_model( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size, net_type ):

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


# -----------------------------------------------------------------------------
# net_type = 1
# Deeper initial feature extraction (two Conv1D blocks) followed by a 3-level
# downsampling/upsampling (encoder-decoder) structure.
# -----------------------------------------------------------------------------
def get_model_1( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 11
# Similar to net_type=1 but with a lighter encoder (one Conv1D before pooling).
# -----------------------------------------------------------------------------
def get_model_11( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 2
# Variant with a different encoder channel schedule and symmetric decoder.
# -----------------------------------------------------------------------------
def get_model_2( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 21
# Like net_type=2 but with fewer early Conv1D layers before downsampling.
# -----------------------------------------------------------------------------
def get_model_21( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 3
# Deeper encoder-decoder with an extra pooling/upsampling level (4 stages).
# -----------------------------------------------------------------------------
def get_model_3( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 31
# Deepest of the 3x family: adds an additional downsampling block before decoding.
# -----------------------------------------------------------------------------
def get_model_31( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 4
# Similar depth to net_type=3 but with different filter scaling in the encoder.
# -----------------------------------------------------------------------------
def get_model_4( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

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


# -----------------------------------------------------------------------------
# net_type = 5
# Aggressive expansion in the decoder (filters multiplied in later layers).
# -----------------------------------------------------------------------------
def get_model_5( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
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


# -----------------------------------------------------------------------------
# net_type = 6
# Uses progressively smaller kernel sizes deeper in the network (kernel_size//2,
# //4, ...), aiming to capture multi-scale features.
# -----------------------------------------------------------------------------
def get_model_6( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
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


# -----------------------------------------------------------------------------
# net_type = 7
# Encoder-only style (no explicit upsampling): stacked Conv/Pool followed by head.
# -----------------------------------------------------------------------------
def get_model_7( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
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


# -----------------------------------------------------------------------------
# net_type = 8
# Wider/deeper variant using an explicit per-layer filter schedule (fil_arr).
# -----------------------------------------------------------------------------
def get_model_8( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
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


# -----------------------------------------------------------------------------
# net_type = 9
# Another fil_arr-based architecture with a distinct depth/width schedule.
# -----------------------------------------------------------------------------
def get_model_9( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
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


# -----------------------------------------------------------------------------
# net_type = 100
# Highly parameterized fil_arr-based encoder-decoder variant.
# -----------------------------------------------------------------------------
def get_model_100( fil_arr, kernel_size, pool_size, dropout_rate, xtrain_size ):

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
