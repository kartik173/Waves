# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:25:18 2019

@author: kartik
"""

import noisereduce as nr
import librosa
import matplotlib.pyplot as plt
import numpy as np


# Plot audio with zoomed in y axis
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()
    
# Load audio file
audio_data, sampling_rate = librosa.load('dataset/fold1/7061-6-0-0.wav')
# Noise reduction
noisy_part = audio_data[0:25000]  
reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
# Visualize
print("Original audio file:")
plotAudio(audio_data)
print("Noise removed audio file:")
plotAudio(reduced_noise)


trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
print("Trimmed audio file:")
plotAudio(trimmed)

stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
stft = np.mean(stft,axis=1)
stft = minMaxNormalize(stft)


plotAudio(stft)

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def predictSound(X):
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts,axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

# ---------------------------- create a neural network model ------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

# define a model
model = nn.Sequential(nn.Linear(257, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

# normalize the data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

# create datasets                              
train_data_dir='dataset/'
test_data_dir='test/'

traindataset = datasets.DatasetFolder(train_data_dir, transform=transform, extensions=('.wav'))
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=64, shuffle=True)

testdataset = datasets.ImageFolder(test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=64, shuffle=True)

import os
labels = []
for files in os.listdir('dataset/'):
    f = files.split()[-1].split('\\')[-1]
    labels.append(f.strip())


#classes=('A','B','C','D','E','F','G','H','I','J')
dataiter = iter(trainloader)
ima, lab = dataiter.next()
print(type(ima))
print(ima.shape)
print(lab.shape)
plt.imshow(ima[1].numpy().squeeze(), cmap='Greys_r')
print(labels[lab[1]])
print(lab[1])

print(len(trainloader))


# train the model
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        #print("y")
        images = images.view(images.shape[0], -1)
        
        # to view an image
        # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        
# test the model
with torch.no_grad():
            model.eval()
            total=0
            matched=0
            l=0
            for images, labels in testloader:
                total+=1
                plt.imshow(images[11].numpy().squeeze(), cmap='Greys_r')
                print(labels[10])
                print(os.listdir('../output/test/'))
                try:
                    ima = images[l].view(1, 784)
                    log_ps = model(ima)
                    ps = torch.exp(logps)
                    a=ps.numpy()
                    m=int(np.where(a==a.max())[1])
                    if int(labels[l])==m:
                        matched+=1
                    print("image with label",int(labels[l]), "matches",m,"with probability", round(a.max(),2))
                    l+=1
                    
                except:
                    l+=1
                    #continue
                

print(total,matched)
