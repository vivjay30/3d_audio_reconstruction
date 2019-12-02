import os

import cv2
import librosa
import numpy as np
from scipy.io import wavfile


def read_file(filename, sample_rate=None, trim=False):
    """
    Reads in a wav file and returns it as an np.float32 array in the range [-1,1]
    """
    file_sr, data = wavfile.read(filename)
    if data.dtype == np.int16:
        data = np.float32(data) / np.iinfo(np.int16).max
    elif data.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))
    if sample_rate is not None and sample_rate != file_sr:
        if len(data) > 0:
            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    if trim and len(data) > 1:
        data = librosa.effects.trim(data, top_db=40)[0]
    return data, file_sr


def log_cqt(fname, sample_rate=None):
    """
    Generates a constant Q transform in dB magnitude
    """
    y, sample_rate = read_file(fname, sample_rate=sample_rate)
    fmin = None
    hop_length = 256
    n_bins = 256
    bins_per_octave = 32
    filter_scale = 0.1

    C = np.abs(librosa.cqt(y, sr=sample_rate, hop_length=hop_length, fmin=fmin,
                           n_bins=n_bins, filter_scale=filter_scale, bins_per_octave=bins_per_octave))
    C_db = librosa.power_to_db(C)
    return C_db


def log_mel_spec_tfm_overlap(fname, sample_rate=None):
    x, sample_rate = read_file(fname, sample_rate=sample_rate)


    # window > full period of a 20hz sinusoid (50ms)
    # 22050 * 0.05 ~= 1102.5
    n_fft = 1120

    # overlap windows by 75%
    # n_fft * (1-0.75)
    hop_length = int(0.25*n_fft)

    # n bins on y_axis
    n_mels = 333

    fmin = 20
    fmax = sample_rate / 2

    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels, power=2.0,
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power)
    return mel_spec_db


def log_mel_spec_tfm(fname, sample_rate=None):
    """
    Generates a mel spectrogram with dB magnitude
    """
    x, sample_rate = read_file(fname, sample_rate=sample_rate)
    
    n_fft = 512
    hop_length = 16
    n_mels = 128  # 128 is better for the direction part
    fmin = 20
    fmax = sample_rate / 2 
    
    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft, 
                                                    hop_length=hop_length, 
                                                    n_mels=n_mels, power=2.0, 
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power)
    return mel_spec_db



def log_mel_spec_original(fname, sample_rate=None):
    y, sample_rate = read_file(fname, sample_rate=sample_rate)
    n_fft = 512
    hop_length = 16
    n_mels = 128
    fmin = 20
    fmax = sample_rate / 2

    original_spectrogram = librosa.stft(y, n_fft=n_fft,
                                        hop_length=hop_length)
    power_spectrogram = np.abs(original_spectrogram) ** 2
    S = librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(S)

    return mel_spec_db, original_spectrogram



def save_spectrogram(spectrogram, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(filename, spectrogram)


def save_mask(masks, dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mask_idx in range(masks.shape[0]):
        mask = masks[mask_idx, :, :]
        cv2.imwrite(os.path.join(dir, "mask{:02d}.png".format(mask_idx)), (mask*255).astype(np.uint8))




