import pandas as pd
import numpy as np
import librosa

from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

class Handler(object):
    
    def __init__(self, csv_file):
        self.df = pd.read_csv('Dataset_idx.csv')
        
        self.IdxListProcess = False
        
    def ReadTable(self):
        return self.df
    
    def IdxProcess(self):
        self.KwIndexs = {}
        
        for ikeys in self.df.keys():
            if ikeys.find('idx') != -1:
                kw_str = ikeys.replace('idx_','')

                x = self.df[ikeys]
                ikw = {}
                for i in range(len(x)):
                    p = x[i]
                    p = p.replace('[','')
                    p = p.replace(']','')

                    p = p.split(' ')

                    if p[0] != '':
                        ikw[i] = list(map(int, p))
                    else:
                        ikw[i] = list(map(int,[-1e9]))                     

                self.KwIndexs[kw_str] = ikw
                
    def ReadIndexs(self, key='All'):
        
        if self.IdxListProcess == False:
            self.IdxProcess()
        
        if key == 'All':
            return self.KwIndexs
        else:
            if [0 for keys in self.KwIndexs.keys() if keys.find(key) != -1] == []:
                raise ValueError('This Key does not exist on Table...')
            
            return self.KwIndexs[key]
    
    def ExtractAudio(self, idx, norm=True):
        wavstr = self.df['file'][idx]  # extrai a string contendo o nome do arquivo de audio
        [_, data_file] = wavread(wavstr) # Lê todo o arquivo de audio

        if norm:
            data_file = data_file/32767 # normaliza as amostras do  áudio para o range [-1,1]
        
        N = data_file.shape[0]      # indica o tamanho do arquivo
        # # quando necessário é plotado o áudio do arquivo 
        # t = np.linspace(0, N/fs, N)
        # plt.plot(t, data_file)   
        
        return data_file, N
    
    def IdxFrameProcess(self, func, rkey='All', ckey='All', **kwargs):
                
        if rkey == 'All':
            dfkey = self.df['file']
        else:
            dfkey = [keys for keys in self.df['file'] if keys.find(rkey) != -1]
        

        if ckey == 'All':
            kwkey = self.KwIndexs
        else:
            kwkey = [self.KwIndexs[keys] for keys in self.KwIndexs.keys() if keys.find(rkey) != -1]
        
        
        frameproc = {}
        self.kwFeat = {}
        for i in range(len(dfkey)):
            data_file, N = self.ExtractAudio(i)

            if self.IdxListProcess == False:
                self.IdxProcess()
            
            self.kwFeat[dfkey[i]] = {}
            for k in kwkey.keys():
                for j in range(len(kwkey[k][i])): # para cada audio, retira os frames kw e as features
                    if kwkey[k][i][j] < 0:
                        break
                        
                    frameproc[j] = func(data_file=data_file, index=kwkey[k][i], **kwargs)
                
                self.kwFeat[dfkey][k] = frameproc
            
        return self.kwFeat

def frameMFCC(data_file, index=0, fs = 0, ofs_mfcc=0, **kwargs):
    kwargs['sr'] = fs
    
    frameSample = data_file[index - frame_lenD2:index + frame_lenD2]
    MFCCsample = librosa.feature.mfcc(y=frameSample, sr=fs, **kwargs)
    
    frameMFCC[j] = MFCCsample[ofs_mfcc:]
    
    return MFCCsample

        