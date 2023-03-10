import librosa
import librosa.display

path = 'C:\\Users\\ramak\\Documents\\Sounds Project\\delme_rec_unlimited_h2tx9qav.wav'
aa,rate = librosa.load(path,sr=48000)
librosa.display.waveplot(aa,rate)