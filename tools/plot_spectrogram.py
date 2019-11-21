import librosa
import librosa.display

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(data_train[0][0][0], ref=np.max)
librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=12000,fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()