#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
aud=pd.read_csv('C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
aud.head()


# In[2]:


aud['relative_path']= '/fold' + aud['fold'].astype(str) + '/' + aud['slice_file_name'].astype(str)
aud1=aud[['relative_path','classID']]
aud1.head()


# In[23]:


import librosa
import librosa.display
import IPython.display as ipd
import math
import matplotlib.pyplot as plt
class AudioUtil():
    dirr='C:\\Users\\ramak\\Downloads\\bass.wav'
    def openn(directory):    
        audd,rate=librosa.load(directory)
        return (audd, rate)
    #print(openn(dirr)[1])
    aud_x=openn(dirr)[0]
    aud_rate=openn(dirr)[1]
    #librosa.display.waveplot(aud_x,aud_rate)
    plt.figure(figsize=(15,5))
    ipd.Audio(aud_x,rate=aud_rate)


# In[17]:


ar1=[1,2,30]
ar2=[ar1,ar1]
print(ar2)


# In[1]:


import librosa
import librosa.display
import IPython.display as ipd
import math
import matplotlib.pyplot as plt
dirr='C:\\Users\\ramak\\Downloads\\bass.wav'
def openn(directory):    
    audd,rate=librosa.load(directory)
    return (audd, rate)
#print(openn(dirr)[1])
aud_x=openn(dirr)[0]
aud_rate=openn(dirr)[1]
librosa.display.waveplot(aud_x,aud_rate)
#plt.figure(figsize=(15,5))
ipd.Audio(aud_x,rate=aud_rate)


# In[125]:


type(aud_x[0])


# In[2]:


aud_x.shape


# In[7]:


type(aud_x)
#import numpy as np
#bass_stereo=np.concatenate((aud_x,aud_x),axis=None)
#np.concatenate((aud_x,aud_x),axis=0).shape
#aud_x.T.shape
#print(aud_x.shape)
aud_x1=aud_x.T
aud_x1


# In[25]:


aud_x.shape


# In[8]:


aud_x.T.shape


# In[2]:


import numpy as np
aud_x_1 = np.array([aud_x])


# In[3]:


aud_x_1.shape


# In[4]:


aud_x_2=aud_x_1.T
aud_x_2.shape


# In[5]:


bass_stereo=np.concatenate((aud_x_2,aud_x_2),axis=1)
bass_stereo.shape
#bass_stereo


# In[6]:


ipd.Audio(bass_stereo,rate=aud_rate)


# In[11]:


bass_stereo_vol=np.concatenate((aud_x_2*2,aud_x_2*0.5),axis=1)
ipd.Audio(bass_stereo_vol,rate=aud_rate)


# In[11]:


#bass_stereo_vol.shape
bass_stt.shape


# In[10]:


bass_stt=np.concatenate((aud_x_1*2,aud_x_1*0.5),axis=0)
ipd.Audio(bass_stt,rate=aud_rate)


# In[21]:


aud_rate


# In[12]:


from IPython.display import Audio
Audio(bass_stt,rate=aud_rate)


# In[13]:


import sklearn
import librosa
import librosa.display
bass_x, bass_rate=librosa.load('C:\\Users\\ramak\\Downloads\\bass.wav')
Audio(bass_x, rate=bass_rate)


# In[16]:


mfcc=librosa.feature.mfcc(bass_x, sr=bass_rate)


# In[26]:


def freqToMel(f):
    return 1127 * math.log(1 + (f/700))

# Vectorize function to apply to numpy arrays
freqToMelv = np.vectorize(freqToMel)

# Observing 0 to 10,000 Hz
Hz = np.linspace(0,1e4) 
# Now we just apply the vectorized function to the Hz variable
Mel = freqToMelv(Hz) 

# Plotting the figure:
fig, ax = plt.subplots(figsize = (20,10))
ax.plot(Hz, Mel)
plt.title('Hertz to Mel')
plt.xlabel('Hertz Scale')
plt.ylabel('Mel Scale')
plt.show()


# In[43]:


print(freqToMel(100))


# In[ ]:


1601.9016624094181 -- 2200
1521.367410001541 -- 2000

999.9907007660177 -- 1000
1125.3419915352404 -- 1200

2578.797292695773 -- 6200
2545.6478440682804 -- 6000

2962.658534787974 -- 9000
2985.659333116345 -- 9200

3519.6142979168626 -- 15200
3505.348284642205 -- 1500

178.31845387718576 -- 120
150.48987948783693 -- 100


# In[36]:


2985.659333116345 - 2962.658534787974


# In[37]:


2578.797292695773 - 2545.6478440682804


# In[38]:


1601.9016624094181 - 1521.367410001541


# In[17]:


import librosa
guit, sr=librosa.load('C:\\Users\\ramak\\Downloads\\guitar.wav', offset=5,duration=1)

librosa.display.waveplot(guit, sr)


# In[88]:


Audio(guit, rate=sr)


# In[18]:


def env_mask(wav, threshold):
    # Absolute value
    wav = np.abs(wav)
    # Point wise mask determination.
    mask = wav > threshold
    return wav[mask]
guit=env_mask(guit, 0.005)


# In[19]:


guit_spec=librosa.feature.melspectrogram(guit)


# In[24]:


type(guit_spec)
guit_spec.shape


# In[26]:


g=librosa.amplitude_to_db(guit_spec)


# In[28]:


g.shape


# In[100]:


i=librosa.display.specshow(g)
plt.title('Mel Spectrogram for Guitar Audio Sample')
plt.colorbar(i)


# In[72]:


bass, srb=librosa.load('C:\\Users\\ramak\\Downloads\\bass.wav', offset=5, duration=5)
librosa.display.waveplot(bass, srb)


# In[73]:


Audio(bass, rate=srb)


# In[76]:


bass_spec_namp=librosa.feature.melspectrogram(bass)
b=librosa.display.specshow(bass_spec_namp)


# In[99]:


bass_1=env_mask(bass,0.005)
bass_spec_db=librosa.feature.melspectrogram(bass_1)
bdb=librosa.amplitude_to_db(bass_spec_db)
b_db=librosa.display.specshow(bdb)
plt.title('Mel Spectrogram for Bass Audio Sample')
plt.colorbar(b_db)


# In[87]:


librosa.display.waveplot(bass_1,srb)


# In[91]:


drums,srd=librosa.load('C:\\Users\\ramak\\Downloads\\drums.wav', offset=5,duration=1)
Audio(drums, rate=srd)


# In[92]:


librosa.display.waveplot(drums, srd)


# In[98]:


drums_1=env_mask(drums, 0.005)
drums_spec=librosa.feature.melspectrogram(drums_1)
drums_specs_todb=librosa.amplitude_to_db(drums_spec)
drum_spectro=librosa.display.specshow(drums_specs_todb)
plt.title('Mel Spectrogram for Drum Audio Sample')
plt.colorbar(drum_spectro)


# In[114]:


guit_mfcc=librosa.feature.mfcc(guit)
gmfcccolor=librosa.display.specshow(guit_mfcc)
plt.colorbar(gmfcccolor)


# In[115]:


bass_mfcc=librosa.feature.mfcc(bass)
bassmfcccolor=librosa.display.specshow(bass_mfcc)
plt.colorbar(bassmfcccolor)


# In[119]:


drum_mfcc=librosa.feature.mfcc(drums)
drummfcccolor=librosa.display.specshow(drum_mfcc)
plt.colorbar(drummfcccolor)


# In[117]:


fig, ax = plt.subplots(1,3, figsize = (20,10))
ax[0].set(title = 'MFCCs of Guitar')
gmfcccolor = librosa.display.specshow(guit_mfcc, x_axis='time', ax=ax[0])
ax[1].set(title = 'MFCCs of Drums')
librosa.display.specshow(drum_mfcc, x_axis='time', ax=ax[1])
ax[2].set(title = 'MFCCs of Bass')
librosa.display.specshow(bass_mfcc,x_axis='time', ax=ax[2])
plt.colorbar(gmfcccolor)


# In[128]:


import torch


# In[ ]:




