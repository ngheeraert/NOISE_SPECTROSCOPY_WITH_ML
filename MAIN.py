from tensorflow.keras import models, optimizers, callbacks, Sequential
#import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import numpy as np
import time
from network_functions import *
from data_generation_functions import generate_final_data

import matplotlib.pyplot as plt 
import matplotlib

#=============================================
#== IMPORTING THE DATA
#=============================================

#== import the data
data_1 = np.load("data/1OverF_1.npz")
data_2 = np.load("data/Lor_1.npz")
data_3 = np.load("data/1OverF-Lorf_1.npz")
c_data = np.append( data_1['arr_0'], np.append( data_2['arr_0'], data_3['arr_0'], axis=0 ), axis=0 )
T_in = data_1['arr_1']         # Time vector for data generation 
s_data = np.append( data_1['arr_2'], np.append( data_2['arr_2'], data_3['arr_2'], axis=0 ), axis=0 )
w0 = data_1['arr_3']           # Omega vector for data generation
T_train = data_1['arr_4']      # Time vector for training data (based on the experimental data)
w_train = data_1['arr_5']      # Omega vector for training data
T2_span = data_1['arr_6']      # T2 distribution
print('-- data loaded')


#== format the data for the training stage
c_train, s_train = \
generate_final_data(c_data,T_in,s_data,w0,T_train,w_train,T2_span)

x_train, x_test, y_train, y_test = train_test_split( c_train, s_train, test_size=0.15)

print('-- data split for training:')
print("  x_train = ",np.shape(x_train))
print("  y_train = ",np.shape(y_train))
print("  x_test = ",np.shape(x_test))
print("  y_test = ",np.shape(y_test))
print("  w0 = ",np.shape(w0))
print("  w_train = ",np.shape(w_train))


#=============================================
#== create the neural net
#=============================================

FILTER_NB=80
KERNEL_SIZE=42
DROPOUT_RATE=0.05
POOL_SIZE=2
X_TRAIN_SIZE = np.shape(x_train)[-1]

model = get_model( filter_nb=FILTER_NB, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE,\
                  dropout_rate=DROPOUT_RATE, xtrain_size=X_TRAIN_SIZE )

#=============================================
#== training
#=============================================

#-- set up the hyperparameters
BATCH_SIZE=64
EPOCHS=500
INIT_LR=1e-3
MIN_LR=1e-6
RED_FACTOR=0.5
DROPOUT_RATE=0.05

#-- define useful filename
filename="CNN_fil="+str(FILTER_NB)+"_ker="+str(KERNEL_SIZE)+"_dr"+str(DROPOUT_RATE)\
            +"_ps="+str(POOL_SIZE)+'_LRini='+str(1e-3)+'_LRmin='+str(1e-6)\
            +'_bs='+str(BATCH_SIZE)+"_ep="+str(EPOCHS)+"_NOISE_TYPES=3"
print("-- filename = "+filename)

#-- prepare the model
model = get_model( filter_nb=FILTER_NB, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE,\
                  dropout_rate=DROPOUT_RATE, xtrain_size=X_TRAIN_SIZE )  #-- create model
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",factor=RED_FACTOR,patience=8,verbose=True,\
    mode="auto",min_delta=0.001,cooldown=0,min_lr=MIN_LR)  #-- define LR reduction strategy
opt = optimizers.Adam(learning_rate=INIT_LR)  #-- define optimizer
model.compile(loss='MAPE', optimizer=opt)  #-- compilation

#-- fit the model
print('-- beginning fit')
t1 = time.time()
history_ = model.fit( x_train, y_train, BATCH_SIZE, epochs=EPOCHS,\
                        validation_data=(x_test, y_test), verbose=True, callbacks=[reduce_lr])
print('-- fit complete')
print('-- time taken=', time.time() - t1)

#-- save the model
model.save('RESULTS/MODEL_'+filename, overwrite=True)

#-- plot the history
plt.plot( np.arange( 0, EPOCHS ) , history_.history['val_loss'] )
plt.savefig( 'RESULTS/VAL_ACC_HISTORY_'+filename+'.pdf', format='pdf' )


#=============================================
#== testting the model
#=============================================

predictions = model.predict(x_test)
#reconstructed_model = models.load_model( 'training/'+filename, compile=False )
#probability_model = Sequential([reconstructed_model])
#probability_model = tf.keras.Sequential([model])

matplotlib.rcParams['figure.dpi']=300

plt.subplot(1, 1, 1)
rand_set = np.random.randint( 0, y_test.shape[0] ,(5,) )
for i in rand_set:
    plt.plot(w_train, y_test[i,:],color='C'+str(i))
    plt.plot(w_train, predictions[i],dashes=[2,2,2,2],color='C'+str(i))
plt.yscale('log')
plt.ylim(1e2,1e5)
plt.xlim(0, 0.5e6)
plt.ylabel('Noise Amplitude')
plt.ylabel('Frequency \omega')
plt.savefig( 'RESULTS/MODEL_TEST_'+filename+'.pdf', format='pdf' )

