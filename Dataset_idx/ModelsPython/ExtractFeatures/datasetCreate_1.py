import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

# # lista de arquivos de palavras OOV

# words_oov = ['one/', 'nine/', 'bed/', 'cat/', 'dog/', 'house/', 'down/']
# for i in range(len(words_oov)):
#     words_oov[i] = '../../' + words_oov[i]

# OOV_words_list = {}

# for i in range(len(words_oov)):
#     OOV_words_list[i] = os.listdir(words_oov[i])

# # lista de arquivos de KWD

# words_kw = ['right/', 'go/', 'left/', 'house/']
# for i in range(len(words_kw)):
#     words_kw[i] = '../../' + words_kw[i]

# KW_words_list = {}

# for i in range(len(words_kw)):
#     KW_words_list[i] = os.listdir(words_kw[i])

df = pd.read_csv('../../Dataset_idx.csv')
# df.head()

# Formata as colunas de indexação em formatos de lista

x = df['idx_abre']
iAbre = {}
for i in range(len(x)):
    p = x[i]
    p = p.replace('[','')
    p = p.replace(']','')
    
    p = p.split(' ')
#     print(p)
    if p[0] != '':
        iAbre[i] = list(map(int, p))
    else:
        iAbre[i] = list(map(int,[-1e9]))

############### DEFINICAO DAS CONSTANTES ###################

# fs = 16000 # taxa de amostragem dos arquivos de audio

# n_fft= 1024   # tamanho da FFT para extração dos MFCCs
# hop_length=0 # pulo entre cada frame
# n_mels= 100   # numero de filtros MEL
# n_mfcc= 18   # numero de coeficientes MFCC
# ofs_mfcc=2   # offset dado para não utilizar os primeiros coeficientes MFCC      
# fmin=100    # frequencia mínima do MFCC
# fmax=8000   # frequencia máxima do MFCC

# n_frames_MFCC = 10 # numero de frames MFCC que será usado para o reconhecimento.

fs = 8000 # taxa de amostragem dos arquivos de audio

n_fft= 512   # tamanho da FFT para extração dos MFCCs
hop_length=0 # pulo entre cada frame
n_mels= 50   # numero de filtros MEL
n_mfcc= 15   # numero de coeficientes MFFC
ofs_mfcc=2   # offset dado para não utilizar os primeiros coeficientes MFCC      

fmin=100    # frequencia mínima do MFCC
fmax=4000   # frequencia máxima do MFCC

n_frames_MFCC = 10 # numero de frames MFCC que será usado para o reconhecimento.

frame_len = (n_frames_MFCC-1)*n_fft # tamanho do frame recortado para cada entrada
frame_lenD2 = int(frame_len/2) # tamanho do frame dividido por 2


############# GERANDO AS FEATURES DAS KEYWORDS #####################

# frameMFCC = {}
# kwFeat = {}
# for i in range(len(words_kw)):
#     for j in range(len(KW_words_list[i])):
#         wavstr = words_kw[i] + KW_words_list[i][j]
#         [_, data_file] = wavread(wavstr) # Lê todo o arquivo de audio

#         data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
#         N = data_file.shape[0]      # indica o tamanho do arquivo
        
#         frameSample = np.zeros(16000)
#         frameSample[:min(N,16000)] = data_file[:min(N,16000)]
#     #     # quando necessário é plotado o áudio do arquivo 
#     #     t = np.linspace(0, N/fs, N)
#     #     plt.plot(t, data_file)
#         MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, fmin=fmin, fmax=fmax, 
#                                                  n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)

#         frameMFCC[j] = MFCCsample[ofs_mfcc:]

#     kwFeat[i] = frameMFCC


frameMFCC = {}
kwFeat = {}
for i in range(len(df['file'])):
    wavstr = df['file'][i]  # extrai a string contendo o nome do arquivo de audio
    [_, data_file] = wavread('../../' + wavstr) # Lê todo o arquivo de audio

    data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
    N = data_file.shape[0]      # indica o tamanho do arquivo

#     # quando necessário é plotado o áudio do arquivo 
#     t = np.linspace(0, N/fs, N)
#     plt.plot(t, data_file)
    for j in range(len(iAbre[i])): # para cada audio, retira os frames kw e as features
        if iAbre[i][j] < 0:
            break
        
        frameSample = data_file[iAbre[i][j]-frame_lenD2:iAbre[i][j]+frame_lenD2]
        MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, fmin=fmin, fmax=fmax, 
                                             n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)

        frameMFCC[j] = MFCCsample[ofs_mfcc:]
    
    kwFeat[i] = frameMFCC


lenKW_1 = 0
for i in range(len(df['file'])):
    if df['file'][i].find('Vitor') == -1:
        for j in range(len(iAbre[i])):
            if iAbre[i][j] < 0:
                break
            
            lenKW_1=lenKW_1+1

lenKW_1_test = 0
for i in range(len(df['file'])):
    if df['file'][i].find('Vitor') != -1:
        for j in range(len(iAbre[i])):
            if iAbre[i][j] < 0:
                break
            
            lenKW_1_test=lenKW_1_test+1
            
print('number of keyword inputs:',lenKW_1)
print('number of keyword test inputs:',lenKW_1_test)
# imgplot = plt.imshow(kwFeat[0][7])


############# GERANDO AS FEATURES DAS OOV WORDS #####################

# frameMFCC = {}
# oovFeat = {}
# for i in range(len(words_oov)):
# # for i in range(2):
#     for j in range(len(OOV_words_list[i])):
#         wavstr = words_oov[i] + OOV_words_list[i][j]
#         [_, data_file] = wavread(wavstr) # Lê todo o arquivo de audio

#         data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
#         N = data_file.shape[0]      # indica o tamanho do arquivo
        
#         frameSample = np.zeros(16000)
#         frameSample[:min(N,16000)] = data_file[:min(N,16000)]
#     #     # quando necessário é plotado o áudio do arquivo 
#     #     t = np.linspace(0, N/fs, N)
#     #     plt.plot(t, data_file)
#         MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, fmin=fmin, fmax=fmax, 
#                                                  n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)

#         frameMFCC[j] = MFCCsample[ofs_mfcc:]

#     oovFeat[i] = frameMFCC

# OOV_lenght = 100 # gera OOV_length OutOfVoc frames por audio
OOV_lenght = list(map(int,df['recOOV'])) # gera OOV_length OutOfVoc frames por audio
iOOV = {}
# iOOVk = np.zeros(OOV_lenght)

for i in range(len(df['file'])):
    wavstr = df['file'][i]  # extrai a string contendo o nome do arquivo de audio
    [_, data_file] = wavread('../../' + wavstr) # Lê todo o arquivo de audio

    data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
    N = data_file.shape[0]      # indica o tamanho do arquivo
#     print(N)
    
    iOOVk = np.zeros(OOV_lenght[i])

    for k in range(OOV_lenght[i]):
        # tenta adquirir indices que indicam frames com Out Of Voc words
        # de forma aleatoria
        goodId = False
        while goodId == False:
            idx = np.random.randint(N-frame_len)+frame_lenD2 # gera numero aleatorio
            goodId = True
            for j in range(len(iAbre[i])): # analisa o indice criado com os indices das kw
                if abs(idx-iAbre[i][j]) < frame_len:
                    goodId = False
        
        iOOVk[k] = idx
            
    iOOV[i] = iOOVk.astype(int)


# i=2
# wavstr = df['file'][i]  # extrai a string contendo o nome do arquivo de audio
# [_, data_file] = wavread(wavstr) # Lê todo o arquivo de audio

# data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
# N = data_file.shape[0]      # indica o tamanho do arquivo

# #     # quando necessário é plotado o áudio do arquivo 
# #     t = np.linspace(0, N/fs, N)
# #     plt.plot(t, data_file)

# frameSample = np.zeros((frame_len,len(iOOV[i])))
# for j in range(len(iOOV[i])): # para cada audio, retira os frames kw e as features
#     frameSample[:,j] = data_file[iOOV[i][j]-frame_lenD2:iOOV[i][j]+frame_lenD2]

# frameSample = frameSample.T.reshape(-1)
# wavwrite('framesOOV.wav', fs, frameSample)

frameMFCC = {}
oovFeat = {}
for i in range(len(df['file'])):
    wavstr = df['file'][i]  # extrai a string contendo o nome do arquivo de audio
    [_, data_file] = wavread('../../' + wavstr) # Lê todo o arquivo de audio

    data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
    N = data_file.shape[0]      # indica o tamanho do arquivo

#     # quando necessário é plotado o áudio do arquivo 
#     t = np.linspace(0, N/fs, N)
#     plt.plot(t, data_file)
    for j in range(len(iOOV[i])): # para cada audio, retira os frames kw e as features
        frameSample = data_file[iOOV[i][j]-frame_lenD2:iOOV[i][j]+frame_lenD2]
        
        MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, fmin=fmin, fmax=fmax, 
                                             n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft)
        
        frameMFCC[j] = MFCCsample[ofs_mfcc:]
    
    oovFeat[i] = frameMFCC


 ################## ANALYSE NUMBER OF KEYWORDS AND OOV #############

# lenOOV = 0
# for i in range(len(words_oov)):
# # for i in range(2):
#     for j in range(len(OOV_words_list[i])):
#         lenOOV=lenOOV+1

# lenKW = 0
# for i in range(len(words_kw)):
#     for j in range(len(KW_words_list[i])):
#         lenKW=lenKW+1
        
# print('number of out-of-voc inputs:', lenOOV)
# print('number of key-word inputs:', lenKW)

lenOOV = 0
for i in range(len(df['file'])):
    if df['file'][i].find('Vitor') == -1:
        for j in range(len(iOOV[i])):
            lenOOV=lenOOV+1

lenOOVtest = 0
for i in range(len(df['file'])):
    if df['file'][i].find('Vitor') != -1:
        for j in range(len(iOOV[i])):
            lenOOVtest=lenOOVtest+1
            
print('number of out-of-voc inputs:', lenOOV)
print('number of out-of-voc test inputs:', lenOOVtest)

# imgplot = plt.imshow(oovFeat[0][99])


############# GERA OS ARRAYS KW E OOV ##############

# # gera os arrays OOV e KW, com suas respectivas labels
# x_OOV = np.zeros((lenOOV, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))
# x_KW = np.zeros((lenKW, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))

# y_OOV = np.zeros((lenOOV,  1))
# y_KW = np.zeros((lenKW,  1))

# ini =0
# fim = len(KW_words_list[0])
# for i in range(len(words_kw)):
#     y_KW[ini:fim] = i+1
#     if i != len(words_kw)-1:
#         ini = fim
#         fim = fim + len(KW_words_list[i+1])

# print(y_KW.shape)


####### funcao para embaralhar e separar os dados em treinamento e teste
# def splitData(x, y, p=0.2): 
#     shapeInX = x.shape
#     shapeInY = y.shape
#     datax = x.reshape(shapeInX[0],-1)
    
#     data = np.concatenate((datax, y), axis=1)
#     # random.randint(0, int(x.shape[0]*(1-p)))
#     xScrambled = np.random.permutation(data)
#     nLines = xScrambled.shape[0]

#     data_train = xScrambled[0:int(round(nLines*(1-p))), :]
#     data_test = xScrambled[int(round(nLines*(1-p))):, :]
# #     print(nLines)
# #     print(data_train.shape)
    
#     shape_train = np.asarray(shapeInX)
#     shape_train[0] = data_train.shape[0]
#     x_train = data_train[:,:data_train.shape[1]-shapeInY[1]].reshape(tuple(shape_train))
#     y_train = data_train[:,[data_train.shape[1]-shapeInY[1]]]
    
    
#     shape_test = np.asarray(shapeInX)
#     shape_test[0] = data_test.shape[0]
#     x_test = data_test[:,:data_test.shape[1]-shapeInY[1]].reshape(tuple(shape_test))
#     y_test = data_test[:,data_test.shape[1]-shapeInY[1]:]
    
    
#     return (x_train, y_train), (x_test, y_test)

def splitData(x, y, p=0.2): # funcao para embaralhar e separar os dados em treinamento e teste
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

# k=0
# for i in range(len(words_oov)):
# # for i in range(2):    
#     for j in range(len(OOV_words_list[i])):
#         x_OOV[k,:,:,0] = oovFeat[i][j]
#         k=k+1

# k=0
# for i in range(len(words_kw)):
#     for j in range(len(KW_words_list[i])):
#         x_KW[k,:,:,0] = kwFeat[i][j]
#         k=k+1


# print('Keyword Abre Array Shape:', x_KW.shape)
# print('Out Of Voc Array Shape:', x_OOV.shape)
        
# x = np.concatenate((x_OOV, x_KW), axis=0)
# y = np.concatenate((y_OOV, y_KW), axis=0)

# print('\nConcatenate Array Shape:', x.shape)

# (x_train, y_train), (x_test, y_test) = splitData(x, y, p=0.2)

# print('\nTrain data shape:', x_train.shape)
# print('Test data shape:', x_test.shape)

# # print(x_train.shape)
# # print(y_train.shape)
# # print(x_test.shape)
# # print(y_test.shape)

# print('\ncounting of Truth keywords in Test Data:', list(y_test[:]).count(1))
# print('counting of OOV words in Test Data:', list(y_test[:]).count(0))

# gera os arrays OOV e KW, com suas respectivas labels
x_OOV = np.zeros((lenOOV, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))
x_KW_1 = np.zeros((lenKW_1, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))

x_OOV_test = np.zeros((lenOOVtest, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))
x_KW_1_test = np.zeros((lenKW_1_test, oovFeat[0][0].shape[0], oovFeat[0][0].shape[1],1))

# y_OOV = np.concatenate((np.ones((lenOOV, 1)), np.zeros((lenOOV,  1))), axis=1)
# y_KW_1 = np.concatenate((np.zeros((lenKW_1,  1)), np.ones((lenKW_1, 1))), axis=1)
y_OOV = np.zeros((lenOOV,  1))
y_KW_1 = np.ones((lenKW_1, 1))

y_OOV_test = np.zeros((lenOOVtest,  1))
y_KW_1_test = np.ones((lenKW_1_test, 1))

k=0
m=0
for i in range(len(df['file'])):
    if df['file'][i].find('Vitor') == -1:
        for j in range(len(iOOV[i])): 
            x_OOV[k,:,:,0] = oovFeat[i][j]
            k=k+1
    else:
        for j in range(len(iOOV[i])): 
            x_OOV_test[m,:,:,0] = oovFeat[i][j]
            m=m+1
        

k=0
m=0
for i in range(len(df['file'])):
    if df['file'][i].find('Vitor') == -1:
        for j in range(len(iAbre[i])):
            if iAbre[i][j] < 0:
                break
            x_KW_1[k,:,:,0] = kwFeat[i][j]
            k=k+1
    else:
        for j in range(len(iAbre[i])): 
            if iAbre[i][j] < 0:
                break
                
            x_KW_1_test[m,:,:,0] = kwFeat[i][j]
            m=m+1

print('Keyword Abre Array Shape:', x_KW_1.shape)
print('Out Of Voc Array Shape:', x_OOV.shape)
        
x = np.concatenate((x_OOV, x_KW_1), axis=0)
y = np.concatenate((y_OOV, y_KW_1), axis=0)
x_t = np.concatenate((x_OOV_test, x_KW_1_test), axis=0)
y_t = np.concatenate((y_OOV_test, y_KW_1_test), axis=0)

print('\nConcatenate Array Shape:', x.shape)

# print(x.shape)
# print(x_OOV.shape)
# print(x_KW_1.shape)
# print(y.shape)
# print(y_OOV.shape)
# print(y_KW_1.shape)

# (x_train, y_train), (x_test, y_test) = splitData(x, y, p=0.1)
(x_train, y_train), _ = splitData(x, y, p=0.0)
(x_test, y_test), _ = splitData(x_t, y_t, p=0.0)

print('\nTrain data shape:', x_train.shape)
print('Test data shape:', x_test.shape)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

print('\ncounting of Truth keywords in Test Data:', list(y_test[:]).count(1))
print('counting of OOV words in Test Data:', list(y_test[:]).count(0))



############################ SAVE THE FEATURE AND VARIABLES ######################

# prepros_feat = [words_oov,
#                 words_kw,
#                 fs,# = 16000 # taxa de amostragem dos arquivos de audio
#                 n_fft,#= 512   # tamanho da FFT para extração dos MFCCs
#                 hop_length,#=0 # pulo entre cada frame
#                 n_mels,#= 50   # numero de filtros MEL
#                 n_mfcc,#= 16   # numero de coeficientes MFCC
#                 ofs_mfcc,#=2   # offset dado para não utilizar os primeiros coeficientes MFCC      
#                 fmin,#=100    # frequencia mínima do MFCC
#                 fmax,#=4000   # frequencia máxima do MFCC
#                 n_frames_MFCC# = 10 # numero de frames MFCC que será usado para o reconhecimento.]
#                 ]
# Rede = {}
# Rede['preFeat'] = prepros_feat
# Rede['x_train'] = x_train
# Rede['y_train'] = y_train
# Rede['x_test'] = x_test
# Rede['y_test'] = y_test

# Rede['x_OOV'] = x_OOV
# Rede['x_KW']  = x_KW
# Rede['y_OOV'] = y_OOV
# Rede['y_KW']  = y_KW

prepros_feat = [fs,# = 16000 # taxa de amostragem dos arquivos de audio
                n_fft,#= 512   # tamanho da FFT para extração dos MFCCs
                hop_length,#=0 # pulo entre cada frame
                n_mels,#= 50   # numero de filtros MEL
                n_mfcc,#= 16   # numero de coeficientes MFCC
                ofs_mfcc,#=2   # offset dado para não utilizar os primeiros coeficientes MFCC      
                fmin,#=100    # frequencia mínima do MFCC
                fmax,#=4000   # frequencia máxima do MFCC
                n_frames_MFCC,# = 10 # numero de frames MFCC que será usado para o reconhecimento.]
                frame_len, #= (n_frames_MFCC-1)*n_fft # tamanho do frame recortado para cada entrada
                frame_lenD2 #= int(frame_len/2) # tamanho do frame dividido por 2
                ]

Rede = {}
Rede['preFeat'] = prepros_feat
Rede['x_train'] = x_train
Rede['y_train'] = y_train
Rede['x_test'] = x_test
Rede['y_test'] = y_test

Rede['x_OOV_train'] = x_OOV
Rede['x_KW_train']  = x_KW_1
Rede['y_OOV_train'] = y_OOV
Rede['y_KW_train']  = y_KW_1

Rede['x_OOV_test'] = x_OOV_test
Rede['x_KW_test']  = x_KW_1_test
Rede['y_OOV_test'] = y_OOV_test
Rede['y_KW_test']  = y_KW_1_test

Rede['x_OOV'] = np.concatenate((x_OOV, x_OOV_test), axis=0)
Rede['x_KW']  = np.concatenate((x_KW_1, x_KW_1_test), axis=0)
Rede['y_OOV'] = np.concatenate((y_OOV, y_OOV_test), axis=0)
Rede['y_KW']  = np.concatenate((y_KW_1, y_KW_1_test), axis=0)

# np.save("DataSet1.npy", Rede)

np.save("../FeaturesSet/DataSet1.npy", Rede)
print('\nDataSet Written!\n\n')
