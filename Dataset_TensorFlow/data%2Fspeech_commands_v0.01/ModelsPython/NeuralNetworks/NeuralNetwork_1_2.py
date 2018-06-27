import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

from librosa.feature import mfcc
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

import os

################ LOAD THE FEATURES AND THE VARIABLES ####################

Rede = np.load("../FeaturesSet/DataSet1.npy").item()

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
from keras import regularizers

input_shape = x_train.shape[1:]
num_classes = len(words_kw) + 1

y_train_t = keras.utils.to_categorical(y_train.reshape(-1), num_classes)
y_test_t = keras.utils.to_categorical(y_test.reshape(-1), num_classes)

filename = 'ModelTrained_1_2_2'

print(y_train.shape)
print(input_shape)
model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 4), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(4, kernel_size=(3, 4), strides=(1, 1),
                 activation='relu'))
                 # kernel_regularizer=regularizers.l2(0.001),
                 # activity_regularizer=regularizers.l1(0.001)))
# model.add(Conv2D(3, kernel_size=(2, 4), strides=(1, 1),
#                  activation='relu'))
# model.add(Conv2D(3, kernel_size=(2, 4), strides=(1, 1),
#                  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(2, kernel_size=(2, 4), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(4, (2, 2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

history = model.fit(x_train, y_train_t,
          validation_split=0.1,
          epochs=50,
          batch_size=128,
          verbose=2)
print('\nNeural Network Trained! \n\n')

########### scores Analyser
print('***** Scores Analyser *****\n\n')

score = model.evaluate(x_test, y_test_t, batch_size=100)
print('Test model with data_test:', 100*score[1],'%\n')


y_KW_t = keras.utils.to_categorical(y_KW.reshape(-1), num_classes)
y_OOV_t = keras.utils.to_categorical(y_OOV.reshape(-1), num_classes)

scoreTrue = model.evaluate(x_KW, y_KW_t)
scoreFalse = model.evaluate(x_OOV, y_OOV_t)


print('Score True of model with entire KW dataset:', 100*scoreTrue[1],'%\n')
print('Score False of model with entire OOV dataset:', 100*scoreFalse[1],'%\n')



model.save('../NeuralNetworkModels/' + filename + '.h5')
Rede['History'] = history.history
Rede['scoreFalse'] = scoreFalse
Rede['scoreTrue']  = scoreTrue
Rede['scoreTest']  = score


np.save('../NeuralNetworkHistories/' + filename + '.npy', Rede)

