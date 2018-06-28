import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# np.set_printoptions(precision=2)

# filename = 'ModelTrained_3_8_10'

# model = load_model('NeuralNetworkModels/' + filename + '.h5')

# Rede = np.load('NeuralNetworkHistories/' + filename + '.npy').item()

# prepros_feat 	= Rede['preFeat']  
# x_train 		= Rede['x_train']
# y_train			= Rede['y_train'] 
# x_test 			= Rede['x_test'] 
# y_test 			= Rede['y_test']

# x_OOV = Rede['x_OOV'] 
# x_KW  = Rede['x_KW'] 
# y_OOV = Rede['y_OOV']
# y_KW  = Rede['y_KW'] 

# words_kw = prepros_feat[1]
# num_classes = len(words_kw) + 1


# y_test_t = keras.utils.to_categorical(y_test.reshape(-1), num_classes)

# # y_hat_test = model.predict(x_test)
# y_hat_test = model.predict_classes(x_test)
# y_hat_test_t = keras.utils.to_categorical(y_hat_test.reshape(-1), num_classes)

# print('shape of y_hat:', y_hat_test.shape)
# print('shape of y:', y_test_t.shape)

# mtx = np.zeros((y_hat_test_t.shape[1],y_hat_test_t.shape[1]))
# for i in range(y_hat_test_t.shape[1]):
# 	tmp = y_hat_test_t[y_test_t[:,i] == 1, :]
# 	line = tmp.sum(axis=0)
# 	top = tmp.sum()

# 	mtx[i,:] = line/top
# 	# print('sum of i:', line/top)

# print('\nConfusion Matrix:')
# print(mtx)


def confusion(model, Rede)
	np.set_printoptions(precision=2)
	prepros_feat 	= Rede['preFeat']  
	x_train 		= Rede['x_train']
	y_train			= Rede['y_train'] 
	x_test 			= Rede['x_test'] 
	y_test 			= Rede['y_test']

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