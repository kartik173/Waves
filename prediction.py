# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:28:22 2019

@author: kartik
"""


import noisereduce as nr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

class Predictions:
    
    # Plot audio with zoomed in y axis
    def plotAudio(self,output):
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
        plt.plot(output, color='blue')
        ax.set_xlim((0, len(output)))
        ax.margins(2, -0.1)
        plt.show()
        
    def minMaxNormalize(self,arr):
        mn = np.min(arr)
        mx = np.max(arr)
        return (arr-mn)/(mx-mn)
    
    def predictSound(self,X):
        stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
        stfts = np.mean(stfts,axis=1)
        stfts = self.minMaxNormalize(stfts)
        #result = model.predict(np.array([stfts]))
        #predictions = [np.argmax(y) for y in result]
        #return lb.inverse_transform([predictions[0]])[0]
        
    def convertAudiotoArray(self,file_path):
        # Load audio file
        audio_data, sampling_rate = librosa.load(file_path)
        # Noise reduction
        noisy_part = audio_data[0:25000]  
        reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)    
        trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
        stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
        stft = np.mean(stft,axis=1)
        stft = self.minMaxNormalize(stft)
        
        return stft
        
        # Visualize
    # =============================================================================
    #     print("Original audio file:")
    #     plotAudio(audio_data)
    #     print("Noise removed audio file:")
    #     plotAudio(reduced_noise)
    #     print("Trimmed audio file:")
    #     plotAudio(trimmed)
    #     plotAudio(stft)
    # =============================================================================
    
    
    def get_Dataframe(self,path):
        arrayList=[]
        for folder in os.listdir(path):
            for file in os.listdir(path+folder):
                f = path+'/'+folder+'/'+file
                array=self.convertAudiotoArray(f)
                if path!='test/':
                    array=np.append(array,folder)
                
                arrayList.append(array.tolist())
                
    #    print(arrayList)
        df=pd.DataFrame(arrayList)
        
        return df
    
    def getAccuracy(self,actual,pred):
        c=0
        for i in range(len(actual)):
            if actual[i]==pred[i]:
                c+=1
        return (c/len(actual))*100
    
    def trainModel(self):
        path='dataset/'
        #labels = os.listdir(path)
        trainData=self.get_Dataframe(path)
        labels, unique=pd.factorize(trainData[257])
        trainData1=pd.concat([trainData.iloc[:,:-1],pd.DataFrame(labels)], axis=1)
        X=trainData1.iloc[:,:-1].values
        Y=trainData1.iloc[:,-1].values
        
        from sklearn.ensemble import RandomForestClassifier
        # Create the model with 100 trees
        model = RandomForestClassifier(n_estimators=50, 
                                       bootstrap = True,
                                       max_features = 'sqrt')
        # Fit on training data
        model.fit(X, Y)
        pickle.dump(model, open('model.pkl', 'wb'))
     
    
    def ValidateModel(self):
        path='validate/'
        #labels = os.listdir(path)
        testData=self.get_Dataframe(path)
        labels, unique=pd.factorize(testData[257])
        testData1=pd.concat([testData.iloc[:,:-1],pd.DataFrame(labels)], axis=1)
        Xval=testData1.iloc[:,:-1].values
        Yval=testData1.iloc[:,-1].values
        
        model = pickle.load(open('model.pkl', 'rb'))
        ypred=model.predict(Xval)
        
        print(self.getAccuracy(Yval, ypred))
        
        #rf_probs = model.predict_proba(Xval)[:, 1]
        
    def testAudio(self):
        path='test/'
        #labels = os.listdir(path)
        testData=self.get_Dataframe(path)
        Xtest=testData.iloc[:,:].values
        
        model = pickle.load(open('model.pkl', 'rb'))
        ypred=model.predict(Xtest)
        
        return ypred[0]

#p=Predictions()
#p.testAudio()

