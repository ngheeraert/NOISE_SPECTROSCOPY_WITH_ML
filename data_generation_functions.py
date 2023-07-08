import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

# For data interpolation
def interpData(x,y,xNew):
	f_interp = interp1d(x,y)
	yNew = f_interp(xNew)
	return yNew

# For preparing training data: Add random noise, then replace low values with zeros
# Run this cell multiple times to generate sets with different random noise but same underlying curves

def prepare_trainData(c_in,T_in,T_train,noiseMax=0.03,cutOff=0.03):
	c_train = interpData(T_in,c_in,T_train)
	for i in range(c_in.shape[0]):
		c_train[i,:] = c_train[i,:] + np.random.normal(0,noiseMax*2/3,size=c_train.shape[1])
		cut = np.squeeze(np.argwhere(c_train[i,:]<=cutOff+np.random.normal(0,noiseMax*2/3,1)))
	if cut.size > 1:
		c_train[i,cut[0]-1:] = 0
	elif cut.size == 1:
		c_train[i,cut-1:] = 0
	return c_train


def generate_final_data(c_data,T_in,s_data,w0,T_train,w_train,T2_span,test_size):
	nnps = 6 #-- noise number per sample
	c_train_1set = prepare_trainData( c_data, T_in, T_train )
	s_train_1set = interpData( w0, s_data, w_train )
	d1 = np.shape( c_train_1set )[0]
	d2 = np.shape( c_train_1set )[1]
	d3 = np.shape( s_train_1set )[1]
	c_train = np.zeros( ( d1*nnps, d2 ) )
	s_train = np.zeros( ( d1*nnps, d3 ) )
	for i in range(nnps):
		c_train_1set = prepare_trainData( c_data, T_in, T_train, noiseMax=0.015,cutOff=0.03 )
		c_train[i*d1:(i+1)*d1,:] = c_train_1set

	return train_test_split( c_train, s_train, test_size=test_size)

