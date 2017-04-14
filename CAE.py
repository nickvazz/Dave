import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model, Sequential
from keras import backend as K
from CAE_input_data import getTempData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data, temps = getTempData()
print data.shape, 'shape'
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
# print input_data
# print K.int_shape(input_data)
encodeConv1 = Convolution3D(32,(2,2,2), padding='same', activation='relu')(input_data)
pool1 = MaxPooling3D(pool_size=(2,2,2), padding='valid')(encodeConv1)
encodeConv2 = Convolution3D(16,(2,2,2), padding='same', activation='relu')(pool1)
pool2 = MaxPooling3D(pool_size=(2,2,2), padding='valid')(encodeConv2)
encodeConv3 = Convolution3D(8,(2,2,2), padding='same', activation='relu')(pool2)
# pool3 = MaxPooling3D(pool_size=(2,2,2), padding='valid')(encodeConv3)
encodeConv4 = Convolution3D(4,(1,1,1), padding='same', activation='relu')(encodeConv3)
# pool4 = MaxPooling3D(pool_size=(2,2,2), padding='valid')

encodingLayer = Dense(2, activation='sigmoid', name='code')(encodeConv4)

upSamp1 = UpSampling3D(size=(2,2,2))(encodingLayer)
decodeCov1 = Convolution3D(4,(2,2,2), padding='same', activation='relu')(upSamp1)
upSamp2 = UpSampling3D(size=(2,2,2))(decodeCov1)
decodeCov2 = Convolution3D(8,(2,2,2), padding='same', activation='relu')(upSamp2)
decodeCov3 = Convolution3D(16,(2,2,2), padding='same', activation='relu')(decodeCov2)
decodeCov4 = Convolution3D(32,(2,2,2), padding='same', activation='relu')(decodeCov3)

output = Convolution3D(200,(2,2,2), padding='same', activation='relu')(decodeCov4)
# print K.int_shape(output)
ConAE = Model(input_data, output)
ConAE.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
ConAE.fit(X_train, X_train, epochs=20, batch_size=15, shuffle=True, validation_data=(X_train,X_train))
# ConAE.fit(dataInTrain, dataInTrain, epochs=5, batch_size=10, shuffle=True, validation_data=(dataInTest,DataInTest))

for layer in ConAE.layers:
    g = layer.get_config() # print gives infor about each layer
    h = layer.get_weights() # print gives weights and bias
    # if g['name'] == 'code':
    #     print 'got it', g, h

middle_layer = Model(inputs=ConAE.input,
                     outputs=ConAE.get_layer('code').output)
middle_layer_output = middle_layer.predict(X_train)
# print middle_layer_output
# print middle_layer_output.shape
# print middle_layer_output.reshape(len(X_train),2), temps

X = middle_layer_output.reshape(len(X_train),2)
X2 = middle_layer_output.reshape(len(X_test),2)

# temps = map(float,temps)
# print len(temps)
# print len(X)
s = plt.scatter(X[:,0], X[:,1], c=y_train)
s2 = plt.scatter(X2[:,0], X2[:,1], c=y_test)
cb = plt.colorbar(s)
cb = plt.colorbar(s2)
cb.set_label('temps')

plt.show()
