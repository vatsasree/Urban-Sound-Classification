import os
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play
from tqdm import tqdm

audio_dataset_path='C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\audio'
metadata_path = 'C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8k.csv'
metadata = pd.read_csv(metadata_path)

new_path = 'C:/Users/ramak/Documents/Datasets/Audio Datasets/Fixed Length Urban Sound 8k Dataset'
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row['fold'])+'/',str(row['slice_file_name']))
    input_file = file_name
    output_file = os.path.join(os.path.abspath(new_path),'fold'+str(row['fold'])+"\\",str(row['slice_file_name']))

    target_file_length = 5*1000
    original_segment = AudioSegment.from_wav(file_name)
    silence_duration = target_file_length - len(original_segment)
    silenced_segment = AudioSegment.silent(duration=silence_duration)
    combined_segment = original_segment + silenced_segment

    combined_segment.export(out_f=output_file, format="wav")
    