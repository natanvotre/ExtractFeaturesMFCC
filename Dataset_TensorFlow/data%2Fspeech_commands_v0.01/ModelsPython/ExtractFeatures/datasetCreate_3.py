import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

from librosa.feature import mfcc
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

import os

# lista de arquivos de palavras OOV

words_oov = ['one/', 'nine/', 'bed/', 'cat/', 'dog/', 'house/', 'down/']
for i in range(len(words_oov)):
    words_oov[i] = '../../' + words_oov[i]

OOV_words_list = {}

for i in range(len(words_oov)):
    OOV_words_list[i] = os.listdir(words_oov[i])

# lista de arquivos de KWD

words_kw = ['right/', 'go/', 'left/', 'house/']
for i in range(len(words_kw)):
    words_kw[i] = '../../' + words_kw[i]

KW_words_list = {}

for i in range(len(words_kw)):
    KW_words_list[i] = os.listdir(words_kw[i])


############### DEFINICAO DAS CONSTANTES ###################

fs = 16000 # taxa de amostragem dos arquivos de audio

n_fft= 512   # tamanho da FFT para extração dos MFCCs
hop_length=0 # pulo entre cada frame
n_mels= 50   # numero de filtros MEL
n_mfcc= 18   # numero de coeficientes MFCC
ofs_mfcc=2   # offset dado para não utilizar os primeiros coeficientes MFCC      
fmin=100    # frequencia mínima do MFCC
fmax=8000   # frequencia máxima do MFCC

n_frames_MFCC = 10 # numero de frames MFCC que será usado para o reconhecimento.

############# GERANDO AS FEATURES DAS KEYWORDS #####################

frameMFCC = {}
kwFeat = {}
for i in range(len(words_kw)):
    for j in range(len(KW_words_list[i])):
        wavstr = words_kw[i] + KW_words_list[i][j]
        [_, data_file] = wavread(wavstr) # Lê todo o arquivo de audio

        data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
        N = data_file.shape[0]      # indica o tamanho do arquivo
        
        frameSample = np.zeros(16000)
        frameSample[:min(N,16000)] = data_file[:min(N,16000)]
    #     # quando necessário é plotado o áudio do arquivo 
    #     t = np.linspace(0, N/fs, N)
    #     plt.plot(t, data_file)
        MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, fmin=fmin, fmax=fmax, 
                                                 n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)

        frameMFCC[j] = MFCCsample[ofs_mfcc:]

    kwFeat[i] = frameMFCC



############# GERANDO AS FEATURES DAS OOV WORDS #####################

frameMFCC = {}
oovFeat = {}
for i in range(len(words_oov)):
# for i in range(2):
    for j in range(len(OOV_words_list[i])):
        wavstr = words_oov[i] + OOV_words_list[i][j]
        [_, data_file] = wavread(wavstr) # Lê todo o arquivo de audio

        data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
        N = data_file.shape[0]      # indica o tamanho do arquivo
        
        frameSample = np.zeros(16000)
        frameSample[:min(N,16000)] = data_file[:min(N,16000)]
    #     # quando necessário é plotado o áudio do arquivo 
    #     t = np.linspace(0, N/fs, N)
    #     plt.plot(t, data_file)
        MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, fmin=fmin, fmax=fmax, 
                                                 n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)

        frameMFCC[j] = MFCCsample[ofs_mfcc:]

    oovFeat[i] = frameMFCC




 ################## ANALYSE NUMBER OF KEYWORDS AND OOV #############

lenOOV = 0
for i in range(len(words_oov)):
# for i in range(2):
    for j in range(len(OOV_words_list[i])):
        lenOOV=lenOOV+1

lenKW = 0
for i in range(len(words_kw)):
    for j in range(len(KW_words_list[i])):
        lenKW=lenKW+1
        
print('number of out-of-voc inputs:', lenOOV)
print('number of key-word inputs:', lenKW)



############# GERA OS ARRAYS KW E OOV ##############

# gera os arrays OOV e KW, com suas respectivas labels
x_OOV = np.zeros((lenOOV, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))
x_KW = np.zeros((lenKW, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))

y_OOV = np.zeros((lenOOV,  1))
y_KW = np.zeros((lenKW,  1))

ini =0
fim = len(KW_words_list[0])
for i in range(len(words_kw)):
    y_KW[ini:fim] = i+1
    if i != len(words_kw)-1:
        ini = fim
        fim = fim + len(KW_words_list[i+1])

print(y_KW.shape)


####### funcao para embaralhar e separar os dados em treinamento e teste
def splitData(x, y, p=0.2): 
    shapeInX = x.shape
    shapeInY = y.shape
    datax = x.reshape(shapeInX[0],-1)
    
    data = np.concatenate((datax, y), axis=1)
    # random.randint(0, int(x.shape[0]*(1-p)))
    xScrambled = np.random.permutation(data)
    nLines = xScrambled.shape[0]

    data_train = xScrambled[0:int(round(nLines*(1-p))), :]
    data_test = xScrambled[int(round(nLines*(1-p))):, :]
#     print(nLines)
#     print(data_train.shape)
    
    shape_train = np.asarray(shapeInX)
    shape_train[0] = data_train.shape[0]
    x_train = data_train[:,:data_train.shape[1]-shapeInY[1]].reshape(tuple(shape_train))
    y_train = data_train[:,[data_train.shape[1]-shapeInY[1]]]
    
    
    shape_test = np.asarray(shapeInX)
    shape_test[0] = data_test.shape[0]
    x_test = data_test[:,:data_test.shape[1]-shapeInY[1]].reshape(tuple(shape_test))
    y_test = data_test[:,data_test.shape[1]-shapeInY[1]:]
    
    
    return (x_train, y_train), (x_test, y_test)



############ SPLIT THE DATA TO TRAIN AND TEST ####################

k=0
for i in range(len(words_oov)):
# for i in range(2):    
    for j in range(len(OOV_words_list[i])):
        x_OOV[k,:,:,0] = oovFeat[i][j]
        k=k+1

k=0
for i in range(len(words_kw)):
    for j in range(len(KW_words_list[i])):
        x_KW[k,:,:,0] = kwFeat[i][j]
        k=k+1


print('Keyword Abre Array Shape:', x_KW.shape)
print('Out Of Voc Array Shape:', x_OOV.shape)
        
x = np.concatenate((x_OOV, x_KW), axis=0)
y = np.concatenate((y_OOV, y_KW), axis=0)

print('\nConcatenate Array Shape:', x.shape)

(x_train, y_train), (x_test, y_test) = splitData(x, y, p=0.2)

print('\nTrain data shape:', x_train.shape)
print('Test data shape:', x_test.shape)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

print('\ncounting of Truth keywords in Test Data:', list(y_test[:]).count(1))
print('counting of OOV words in Test Data:', list(y_test[:]).count(0))



############################ SAVE THE FEATURE AND VARIABLES ######################

prepros_feat = [words_oov,
                words_kw,
                fs,# = 16000 # taxa de amostragem dos arquivos de audio
                n_fft,#= 512   # tamanho da FFT para extração dos MFCCs
                hop_length,#=0 # pulo entre cada frame
                n_mels,#= 50   # numero de filtros MEL
                n_mfcc,#= 16   # numero de coeficientes MFCC
                ofs_mfcc,#=2   # offset dado para não utilizar os primeiros coeficientes MFCC      
                fmin,#=100    # frequencia mínima do MFCC
                fmax,#=4000   # frequencia máxima do MFCC
                n_frames_MFCC# = 10 # numero de frames MFCC que será usado para o reconhecimento.]
                ]
Rede = {}
Rede['preFeat'] = prepros_feat
Rede['x_train'] = x_train
Rede['y_train'] = y_train
Rede['x_test'] = x_test
Rede['y_test'] = y_test

Rede['x_OOV'] = x_OOV
Rede['x_KW']  = x_KW
Rede['y_OOV'] = y_OOV
Rede['y_KW']  = y_KW

np.save("../FeaturesSet/DataSet3.npy", Rede)
print('DataSet Written')

plt.imshow(kwFeat[0][10])
plt.show()