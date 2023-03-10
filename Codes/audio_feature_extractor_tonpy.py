import numpy as np
import pandas as pd
import librosa
import librosa.display
import os
from tqdm import tqdm
import math, random
import torch
import torchaudio
from torchaudio import transforms

audio_dataset_path = 'C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\audio'
metadata = pd.read_csv('C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8K.csv')


# def fix_audio_len(file_name):
#     audio, sample_rate = librosa.load(file_name,sr=48000 ,res_type='kaiser_fast')
#     audio_len = 5
#     if audio.shape[0] < (audio_len * sample_rate):
#         audio = np.pad(audio,int(np.ceil((audio_len * sample_rate - audio.shape[0])/2)), mode = 'reflect')
#     audio = audio[:audio_len * sample_rate]
#     return audio    


def fix_audio_len(file_name):
    audio, sample_rate = librosa.load(file_name, sr=48000,res_type='kaiser_fast')
    audio_len = 5
    if audio.shape[0] < (audio_len * sample_rate):
        audio = np.pad(audio,(0,int((audio_len * sample_rate - len(audio)))),mode='constant')
    audio = audio[:audio_len * sample_rate]
    return audio 


# def fix_audio_len_torch(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)



def feature_extractor_mfcc1d(file_name):
    audio, sample_rate = librosa.load(file_name,sr = 48000, res_type='kaiser_fast')
    normalized_y = librosa.util.normalize(audio)
    # print(normalized_y)
    mfccs_feature = librosa.feature.mfcc(normalized_y, sample_rate, n_mfcc = 40)
    mfcc_scaled = np.mean(mfccs_feature.T, axis=0)
    return mfcc_scaled
    # return mfccs_feature



def feature_extractor_mfcc2d(file_name):
    audio, sample_rate = librosa.load(file_name, sr=48000,res_type='kaiser_fast')
    # audio_len = len(audio)//sample_rate
    audio_fixed = fix_audio_len(file_name)
    normalized_y = librosa.util.normalize(audio_fixed)
    # print(normalized_y)
    mfccs_feature = librosa.feature.mfcc(normalized_y, sample_rate, n_mfcc = 40)
    # mfcc_scaled = np.mean(mfccs_feature.T, axis=0)
    return mfccs_feature



def feature_extractor_melspec(file_name):
    
    n_fft=2048
    hop_length=512
    n_mels = 128
    audio, sample_rate = librosa.load(file_name, sr=48000,res_type='kaiser_fast')
    normalized_y = librosa.util.normalize(audio) #normalizing between -1 and 1
    stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(S=stft,sr=48000, n_mels=n_mels)

    # Convert sound intensity to log amplitude:
    mel_db = librosa.amplitude_to_db(abs(mel))
    mfcc_scaled = np.mean(mel_db.T, axis=0) #converting to 1D


    # Normalize between -1 and 1
    normalized_mel = librosa.util.normalize(mel_db)

    return mfcc_scaled


def feature_extractor_melspectrogram_latest(file_name):
    n_fft= 2048
    hop_length=512
    n_mels = 128    
    normalized = librosa.util.normalize(fix_audio_len(file_name))
    stft = librosa.core.stft(normalized,n_fft=n_fft,hop_length=hop_length)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)
    mel_db = librosa.amplitude_to_db(abs(mel))
    normalized_mel = librosa.util.normalize(mel_db)
    return normalized_mel









#=====================================================================================================================================================

def extract_features_and_save_as_npy():
    extracted_features=[]
    for index_num, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row['fold'])+'\\',str(row['slice_file_name']))
        final_class_labels = row['class']
        data  = feature_extractor_melspec(file_name)
        extracted_features.append([data, final_class_labels])  

    extracted_features_melspec_np = np.array(extracted_features)
    np_direc = 'C:\\Users\\ramak\\Documents\\Sounds Project\\npy\\extracted_features_melspec_np.npy'
    np.save(np_direc,extracted_features_melspec_np)


def extract_features_mfcc2d_and_save_as_npy():
    extracted_features=[]
    for index_num, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row['fold'])+'\\',str(row['slice_file_name']))
        final_class_labels = row['class']
        data  = feature_extractor_mfcc2d(file_name)
        extracted_features.append([data, final_class_labels])  

    extracted_features_melspec_np = np.array(extracted_features)
    np_direc = 'C:\\Users\\ramak\\Documents\\Sounds Project\\npy\\extracted_features_mfcc2d_len_fixed_np.npy'
    np.save(np_direc,extracted_features_melspec_np)



def extract_features_melspectrogram_and_save_as_npy_LATEST():
    extracted_features=[]
    for index_num, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row['fold'])+'\\',str(row['slice_file_name']))
        final_class_labels = row['class']
        data  = feature_extractor_melspectrogram_latest(file_name)
        extracted_features.append([data, final_class_labels])  

    extracted_features_melspec_np = np.array(extracted_features)
    np_direc = 'C:\\Users\\ramak\\Documents\\Sounds Project\\npy\\melspectrogram_LATEST_extracted_features_len_fixed_np.npy'
    np.save(np_direc,extracted_features_melspec_np)




# data3 = feature_extractor_mfcc2d('C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\Mini Urban Sound8K\\Class 6\\7062-6-0-0.wav')
# dataa = feature_extractor_mfcc2d('C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\Mini Urban Sound8K\\Class 9\\6508-9-0-6.wav')
# print(data3.shape,dataa.shape)    

# print(len(fix_audio_len('C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\Mini Urban Sound8K\\Class 3\\4912-3-1-0.wav')))

# extract_fixed_length_features = np.load(r'C:\\Users\\ramak\\Documents\\Sounds Project\\npy\\extracted_features_mfcc2d_len_fixed_np.npy',allow_pickle=True)
# print(extract_fixed_length_features[0][0].shape)

# data1 = feature_extractor_melspectrogram_latest('C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\Mini Urban Sound8K\\Class 6\\7062-6-0-0.wav')
# print(data1.shape)


extract_features_melspectrogram_and_save_as_npy_LATEST()    
