import glob
import json
import os

import numpy as np
import torch
import librosa


class SpatialAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(SpatialAudioDataset, self).__init__()

        # Data is stored in subdirectories. Get all of them
        self.dirs = sorted(glob.glob(os.path.join(data_dir, '*')))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        curr_dir = self.dirs[idx]

        # Get all WAV files in subdirectory
        audio_files = sorted(glob.glob(os.path.join(curr_dir, "*.wav")))

        # First load data
        # Todo: Make better spectrogram with dimensions, shape,
        # Maybe look into wavelets here or Mel Cepstrum
        specgrams = []
        for audio_file in audio_files:
            waveform, sr = librosa.load(audio_file, sr=12000)
            # waveform, sr = librosa.load(audio_file)
            specgram = librosa.feature.melspectrogram(\
            y=waveform, sr=sr, n_fft=1024, hop_length=565)
            specgrams.append(torch.from_numpy(specgram))

        data = torch.stack(specgrams) # NUM_MICS x Freq_bins x Time_bins

        # Now load labels
        with open(os.path.join(curr_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # Get the direction in radians from -pi to pi
        position = metadata["source1"]  # x,y,z
        direction = np.arctan(position[1] / position[0])
        if position[0] < 0 and position[1] < 0:
            direction -= np.pi

        elif position[0] < 0 and position[1] > 0:
            direction += np.pi

        return data, torch.tensor([direction])
