import glob
import json
import os
import sys

import numpy as np
import torch
import librosa

import math

sys.path.insert(1, os.path.realpath(os.path.pardir))
from d3audiorecon.tools.utils import read_file, log_mel_spec_tfm, \
    save_spectrogram

NUM_BINS = 12


class SpatialAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(SpatialAudioDataset, self).__init__()

        # Data is stored in subdirectories. Get all of them
        self.dirs = sorted(glob.glob(os.path.join(data_dir, '*')))[:1000]
        self.cache = {}

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if idx in self.cache:
            print("skipped!")
            return self.cache[idx]

        curr_dir = self.dirs[idx]

        # Get all WAV files in subdirectory
        mixed_audio_files = sorted(glob.glob(os.path.join(curr_dir, "*_mixed.wav")))
        #gt_audio_files = sorted(glob.glob(os.path.join(curr_dir, "*_source00_gt.wav")))

        # First load data
        # Todo: Make better spectrogram with dimensions, shape,
        # Maybe look into wavelets here or Mel Cepstrum
        mixed_specgrams = []
        for i, mixed_audio_file in enumerate(mixed_audio_files):
            specgram = log_mel_spec_tfm(mixed_audio_file, sample_rate=22500)
            # save_spectrogram(specgram, "spec_{}.png".format(i))
            # waveform, sr = librosa.load(mixed_audio_file, sr=24000)
            # specgram = librosa.feature.melspectrogram(y=waveform, sr=sr)
            # specgram = librosa.feature.melspectrogram(\
            # y=waveform, sr=sr, n_fft=1024, hop_length=565)
            mixed_specgrams.append(torch.from_numpy(specgram))

        mixed_data = torch.stack(mixed_specgrams) # NUM_MICS x Freq_bins x Time_bins

        # Now load labels
        with open(os.path.join(curr_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # Get the direction in radians from -pi to pi
        position = metadata["source00"]["position"]  # x,y,z
        angular_direction = np.arctan2(position[1], position[0])
        label = int(math.floor((angular_direction + np.pi) * NUM_BINS / (2 * np.pi)))

        # Ground truth spec
        # gt_specgrams = []
        # for gt_audio_file in gt_audio_files:
        #     waveform, sr = librosa.load(gt_audio_file, sr=12000)
        #     specgram = librosa.feature.melspectrogram(\
        #     y=waveform, sr=sr, n_fft=1024, hop_length=565)
        #     gt_specgrams.append(torch.from_numpy(specgram))

        # gt_data = torch.stack(gt_specgrams) # NUM_MICS x Freq_bins x Time_bins

        # Generate GT mask for only for source0 (foreground voice)
        masks = None # torch.div(gt_data, mixed_data) # hope I don't divide by zero :)

        return (mixed_data, torch.tensor(label))
        # return self.cache[idx]#, masks
