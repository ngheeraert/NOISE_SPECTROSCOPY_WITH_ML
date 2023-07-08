from tensorflow.keras import models, layers

def get_model( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
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
