import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model, Sequential
import keras.optimizers
from keras import backend as K
from CAE_input_data import getTempData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import pandas as pd
import os

config = tf.ConfigProto().gpu_options.per_process_gpu_memory_fraction = 0.4

data, temps = getTempData()
# print data.shape, 'shape'
data = data.reshape(len(data)/12800,12800)
data = np.asarray([item.reshape(4,4,4,200).swapaxes(1,2).swapaxes(1,2) for item in data])
print data.shape
# data = np.asarray(data.reshape(20,4,4,4,200))
# print data.T.shape
# data = np.asarray(data.reshape(20,4,4,4,200)) # shape required


# temps = np.asarray([[t]*2 for t in temps]).ravel()
X_train, X_test, y_train, y_test = train_test_split(data, temps, test_size=.3, random_state=42)
# print X_train.shape

# model = Sequential()

input_data = Input(shape=(4,4,4,200,))
encodeConv1 = Convolution3D(32,(2,2,2), padding='same', activation='relu')(input_data)
pool1 = MaxPooling3D(pool_size=(2,2,2), padding='valid')(encodeConv1)
encodeConv2 = Convolution3D(16,(2,2,2), padding='same', activation='relu')(pool1)
pool2 = MaxPooling3D(pool_size=(2,2,2), padding='valid')(encodeConv2)
encodeConv3 = Convolution3D(8,(1,1,1), padding='same', activation='relu')(pool2)

encodingLayer = Dense(2, activation='sigmoid', name='code')(encodeConv3)

decodeCov1 = Convolution3D(8,(1,1,1), padding='same', activation='relu')(encodingLayer)
upSame1 = UpSampling3D(size=(2,2,2))(decodeCov1)
decodeCov2 = Convolution3D(16,(2,2,2), padding='same', activation='relu')(upSame1)
upSamp2 = UpSampling3D(size=(2,2,2))(decodeCov2)
decodeCov3 = Convolution3D(32,(2,2,2), padding='same', activation='relu')(upSamp2)

output = Convolution3D(200,(2,2,2), padding='same', activation='relu')(decodeCov3)

# print K.int_shape(output)
ConAE = Model(input_data, output)

# adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = keras.optimizers.SGD(lr=0.01)

# rms = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.0)


ConAE.compile(optimizer=sgd, loss='mse', metrics=['mae'])
# ConAE.fit(X_train, X_train, epochs=10, batch_size=10, shuffle=True, validation_split=0.1, validation_data=(X_train,X_train))
ConAE.fit(X_train, X_train, epochs=80, batch_size=10, shuffle=True,callbacks=[TensorBoard(log_dir='/Users/Nick/Desktop/research/Dave/tmp/autoencoder')]) #, validation_split=0.1, validation_data=(X_train,X_train))
# ConAE.fit(X_train, X_train, epochs=2, batch_size=10, shuffle=True,callbacks=[TensorBoard(log_dir='/Users/Nick/Desktop/research/Dave/tmp/autoencoder')]) #, validation_split=0.1, validation_data=(X_train,X_train))

for layer in ConAE.layers:
    g = layer.get_config() # print gives infor about each layer
    h = layer.get_weights() # print gives weights and bias
    # if g['name'] == 'code':
    #     print 'got it', g, h

middle_layer = Model(inputs=ConAE.input,
                     outputs=ConAE.get_layer('code').output)
middle_layer_output = middle_layer.predict(X_train)

middle_layer2 = Model(inputs=ConAE.input,
                     outputs=ConAE.get_layer('code').output)
middle_layer_output2 = middle_layer2.predict(X_test)

# print middle_layer_output
# print middle_layer_output.shape
# print middle_layer_output.reshape(len(X_train),2), temps

X = middle_layer_output.reshape(len(X_train),2)
X2 = middle_layer_output2.reshape(len(X_test),2)

# temps = map(float,temps)
# print len(temps)
# print len(X)
# s = plt.scatter(X[:,0], X[:,1], c=y_train,s=10)
# s2 = plt.scatter(X2[:,0], X2[:,1], c=y_test,s=10)
# cb = plt.colorbar(s)

df = pd.DataFrame({ 'x':np.append(X[:,0],X2[:,0]),
                    'y':np.append(X[:,1],X2[:,1]),
                    'T':np.append(y_train,y_test)})

df.to_csv('dataCAE.csv', sep=',')
# cb = plt.colorbar(s2)
# cb.set_label('temps')

from playsound import playsound
playsound('goat.wav')
playsound('goat2.wav')

# plt.show()
os.system('python CAEembedding.py')
