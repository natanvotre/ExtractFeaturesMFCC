import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

from librosa.feature import mfcc
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

import os
# import confusion
def confusion(model, Rede):
  np.set_printoptions(precision=2)
  prepros_feat  = Rede['preFeat']  
  x_train     = Rede['x_train']
  y_train     = Rede['y_train'] 
  x_test      = Rede['x_test'] 
  y_test      = Rede['y_test']

  x_OOV = Rede['x_OOV'] 
  x_KW  = Rede['x_KW'] 
  y_OOV = Rede['y_OOV']
  y_KW  = Rede['y_KW'] 

  words_kw = prepros_feat[1]
  num_classes = len(words_kw) + 1


  y_test_t = keras.utils.to_categorical(y_test.reshape(-1), num_classes)

  # y_hat_test = model.predict(x_test)
  y_hat_test = model.predict_classes(x_test)
  y_hat_test_t = keras.utils.to_categorical(y_hat_test.reshape(-1), num_classes)

  # print('shape of y_hat:', y_hat_test.shape)
  # print('shape of y:', y_test_t.shape)

  mtx = np.zeros((y_hat_test_t.shape[1],y_hat_test_t.shape[1]))
  for i in range(y_hat_test_t.shape[1]):
    tmp = y_hat_test_t[y_test_t[:,i] == 1, :]
    line = tmp.sum(axis=0)
    top = tmp.sum()

    mtx[i,:] = line/top
    # print('sum of i:', line/top)

  print('\nConfusion Matrix:')
  print(mtx)
  return(mtx)



################ LOAD THE FEATURES AND THE VARIABLES ####################
datasetN = '3'
NeuralNumber = '10'

InfoNNs = np.load('../InfoNNs.npy').item()
InfoNNs[datasetN + '_' + NeuralNumber] = InfoNNs[datasetN + '_' + NeuralNumber] + 1

NeuralLoad = str(InfoNNs[datasetN + '_' + NeuralNumber])


Rede = np.load("../FeaturesSet/DataSet" + datasetN + ".npy").item()

# prepros_feat 	= Rede['preFeat']  
# x_train 		= Rede['x_train']
# y_train			= Rede['y_train'] 
# x_test 			= Rede['x_test'] 
# y_test 			= Rede['y_test']
prepros_feat 	= Rede['preFeat']  
x_train 		= Rede['x_train']
y_train			= Rede['y_train'] 
x_test 			= Rede['x_test'] 
y_test 			= Rede['y_test']

x_OOV = Rede['x_OOV'] 
x_KW  = Rede['x_KW'] 
y_OOV = Rede['y_OOV']
y_KW  = Rede['y_KW'] 


words_oov = prepros_feat[0]
words_kw = prepros_feat[1]
fs = prepros_feat[2] # = 16000 # taxa de amostragem dos arquivos de audio
n_fft = prepros_feat[3] #= 512   # tamanho da FFT para extração dos MFCCs
hop_length = prepros_feat[4] #=0 # pulo entre cada frame
n_mels = prepros_feat[5] #= 50   # numero de filtros MEL
n_mfcc = prepros_feat[6] #= 16   # numero de coeficientes MFCC
ofs_mfcc = prepros_feat[7] #=2   # offset dado para não utilizar os primeiros coeficientes MFCC      
fmin = prepros_feat[8] #=100    # frequencia mínima do MFCC
fmax = prepros_feat[9] #=4000   # frequencia máxima do MFCC
n_frames_MFCC = prepros_feat[10] # = 10 # numero de frames MFCC que será usado para o reconhecimento.]

print('Features loaded..\n')

######################## Neural Network Train #######################

print('Initiate Train..\n\n')

import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Sequential

input_shape = x_train.shape[1:]
num_classes = len(words_kw) + 1

y_train_t = keras.utils.to_categorical(y_train.reshape(-1), num_classes)
y_test_t = keras.utils.to_categorical(y_test.reshape(-1), num_classes)

print(y_train.shape)
print(input_shape)

y_train_t_1 = y_train_t[y_train[:,0]<=1, :]
x_train_t_1 = x_train[y_train[:,0]<=1, :, :, :]

filename = 'ModelTrained_' + datasetN + '_' + NeuralNumber + '_' + NeuralLoad

model = Sequential()
# keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
model.add(Conv2D(6, kernel_size=(input_shape[0]-5, 10), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(Dropout(0.3))
model.add(Conv2D(4, kernel_size=(2, 6), strides=(1, 1),
                 activation='relu'))
                 # kernel_regularizer=regularizers.l2(0.01),
                 # activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.3))
# model.add(Conv2D(6, kernel_size=(2, 6), strides=(1, 1),
#                  activation='relu'))
# model.add(Conv2D(5, kernel_size=(2, 4), strides=(1, 1),
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(3, kernel_size=(2, 4), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(4, (2, 2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

monitor = keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=-0.03,
                              patience=0,
                              verbose=0, mode='auto')
callbacks = [early_stop]

# history = model.fit(x_train, y_train_t,
#           validation_split=0.1,
#           epochs=500,
#           batch_size=128,
#           verbose=2,
#           callbacks=callbacks)
history = model.fit(x_train_t_1, y_train_t_1,
          validation_split=0.1,
          epochs=500,
          batch_size=128,
          verbose=2,
          callbacks=callbacks)
print('\nNeural Network Trained! \n\n')

########### scores Analyser
print('***** Scores Analyser *****\n\n')

score = model.evaluate(x_test, y_test_t, batch_size=100, verbose=2)
print('Test model with data_test:', 100*score[1],'%\n')


# y_KW_t = keras.utils.to_categorical(y_KW.reshape(-1), num_classes)
# y_OOV_t = keras.utils.to_categorical(y_OOV.reshape(-1), num_classes)

# scoreTrue = model.evaluate(x_KW[y_KW_t[:,0]==1,:,:,:], y_KW_t[y_KW_t[:,0]==1,:], verbose=2)
# scoreTrue = model.evaluate(x_KW, y_KW_t, verbose=2)
# scoreTrue = model.evaluate(x_KW, y_KW_t, verbose=2)
# scoreTrue = model.evaluate(x_KW, y_KW_t, verbose=2)
# scoreFalse = model.evaluate(x_OOV, y_OOV_t, verbose=2)


# print('Score True of model with entire KW dataset:', 100*scoreTrue[1],'%\n')
# print('Score False of model with entire OOV dataset:', 100*scoreFalse[1],'%\n')

confusion(model, Rede)


model.save('../NeuralNetworkModels/' + filename + '.h5')
Rede['History'] = history.history
Rede['scoreFalse'] = scoreFalse
Rede['scoreTrue']  = scoreTrue
Rede['scoreTest']  = score


np.save('../NeuralNetworkHistories/' + filename + '.npy', Rede)

print('saved name: \'' + filename + '\'')

model.summary()

np.save('../InfoNNs.npy', InfoNNs)
