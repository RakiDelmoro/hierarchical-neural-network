import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./datasets/UrbanSound8K/UrbanSound8K.csv")

# CSV Table
# print(dataset.head())

plt.figure(figsize=(20, 10))
audio_array, sr = librosa.load('./datasets/UrbanSound8K/audio/fold1/7061-6-0-0.wav')
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()


#TODO prepare datasets for training!


