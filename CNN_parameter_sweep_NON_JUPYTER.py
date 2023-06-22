from tensorflow.keras import models, layers, optimizers, callbacks, backend
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import time
import pickle
import sys

def get_model( filter_nb, kernel_size, pool_size, learning_rate, dropout_rate ):
	model = models.Sequential()
	model.add( layers.Input( shape=(x_train.shape[-1], 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D(filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(1,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	opt = optimizers.Adam(learning_rate=learning_rate)
	#model.compile(loss='mean_squared_error', optimizer=opt)
	model.compile(loss='MAPE', optimizer=opt)

	return model



if __name__ == "__main__":
	
	#-- loading the date
	w0 = np.loadtxt( "frequencies.txt" )
	x_train = np.loadtxt("c_train_1_over_f_2.txt")
	y_train = np.loadtxt("s_train_1_over_f_2.txt")
	x_test = np.loadtxt("c_test_1_over_f_2.txt")
	y_test = np.loadtxt("s_test_1_over_f_2.txt")
	print("x_train = ",np.shape(x_train))
	print("y_train = ",np.shape(y_train))
	print("x_test = ",np.shape(x_test))
	print("y_test = ",np.shape(y_test))
	print("w0 = ",np.shape(w0))

	#-- setting the sweep parameters
	label1 = 'number_of_filters'
	label2 = 'kernel_size'
	#parameters1_arr = np.arange( 20,80 )  #-- number of filters
	#parameters2_arr = np.arange( 12,48 )  #-- kernel size
	parameters1_arr = np.arange( 70,80 )  #-- number of filters
	parameters2_arr = np.arange( 40,48 )  #-- kernel size
	pairs = []
	for p1 in parameters1_arr:
		for p2 in parameters2_arr:
			pairs.append([p1,p2])

	training_losses = {}
	validation_losses = {}
	nb_of_points = len(parameters1_arr)*len(parameters2_arr)
	print(label1, parameters1_arr)
	print(label2, parameters2_arr)
	print(len(parameters1_arr),len(parameters2_arr),nb_of_points)


	t1=time.time()
	print('-- nb of points =',nb_of_points )
	from datetime import datetime
	now = datetime.now()
	t_string = now.strftime("%d_%m||%Hh%M")
	
	if (nb_of_points != len(pairs)):
		print("-- ABORT: parameter sweep already started.")
		sys.exit()
	
	count=0
	epoch_nb = 2
	best_loss=1000
	#-- sweep start
	while pairs:
		count +=1
		rand_int = np.random.randint(0, len(pairs))
		pair = pairs[rand_int]
		p1 = int( pair[0] )
		p2 = int( pair[1] )
		pairs.remove(pair)
	
		print('.',end=' ' )
	
		reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=5,verbose=0,\
				mode="auto",min_delta=0.001,cooldown=0,min_lr=1e-5)
		model = get_model( filter_nb=p1, kernel_size=p2, pool_size=2,\
				learning_rate=0.001, dropout_rate=0.05 )
		history_ = model.fit( x_train, y_train, 64, epochs=epoch_nb,\
				validation_data=(x_test, y_test), verbose=0)#, callbacks=[reduce_lr])
	
		training_losses[ p1 , p2 ] = history_.history['loss'][-1]
		validation_losses[ p1 ,p2 ] = history_.history['val_loss'][-1]
		if count > 0:
			with open('training/training_losses_'+t_string+'.pkl', 'wb') as f1:
				pickle.dump(training_losses, f1)
			with open('training/validation_losses_'+t_string+'.pkl', 'wb') as f2:
				pickle.dump(validation_losses, f2)
	
		if (validation_losses[ p1 , p2 ] < best_loss):
			if count > 0:
				with open('training/best_history_'+t_string+'.pkl', 'wb') as f3:
					pickle.dump(history_.history, f3)
			best_loss = validation_losses[ p1 , p2 ]
			model.save('training/best_model'+t_string)
			best_p1=p1
			best_p2=p2
	
		#backend.tensorflow_backend._SESSION.close()
		#backend.tensorflow_backend._SESSION = None
		#backend.clear_session()
		#del model 
	
		if count>0:
			break
	
	
	t2=time.time()
	print(' ')
	print('=== COMPLETE ===')
	print("-- total time = ", t2-t1)
	print("-- best {}, {} =".format(label1, label2), best_p1, best_p2)
	print("-- t_string", t_string)

