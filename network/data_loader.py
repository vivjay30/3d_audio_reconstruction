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
        mixed_audio_files = sorted(glob.glob(os.path.join(curr_dir, "*mixed.wav")))
        gt_audio_files = sorted(glob.glob(os.path.join(curr_dir, "*source_00_gt.wav")))
        print(gt_audio_files)

        # First load data
        # Todo: Make better spectrogram with dimensions, shape,
        # Maybe look into wavelets here or Mel Cepstrum
        mixed_specgrams = []
        for mixed_audio_file in mixed_audio_files:
            waveform, sr = librosa.load(mixed_audio_file, sr=12000)
            specgram = librosa.feature.melspectrogram(\
            y=waveform, sr=sr, n_fft=1024, hop_length=565)
            mixed_specgrams.append(torch.from_numpy(specgram))

        mixed_data = torch.stack(mixed_specgrams) # NUM_MICS x Freq_bins x Time_bins


        # Now load labels
        with open(os.path.join(curr_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # Get the direction in radians from -pi to pi
        position = metadata["source00"]["position"]  # x,y,z
        angular_direction = np.arctan2(position[1], position[0])
        
        # Ground truth spec
        gt_specgrams = []
        for gt_audio_file in gt_audio_files:
            waveform, sr = librosa.load(gt_audio_file, sr=12000)
            specgram = librosa.feature.melspectrogram(\
            y=waveform, sr=sr, n_fft=1024, hop_length=565)
            gt_specgrams.append(torch.from_numpy(specgram))

        gt_data = torch.stack(gt_specgrams) # NUM_MICS x Freq_bins x Time_bins

        # Generate GT mask for only for source0 (foreground voice)
        masks = torch.div(gt_data, mixed_data) # hope I don't divide by zero :)

        return mixed_data, torch.tensor([angular_direction]), masks
