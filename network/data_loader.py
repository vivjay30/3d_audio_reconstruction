import glob
import json
import os
import sys
import math

import numpy as np
import torch
import librosa

from d3audiorecon.tools.utils import read_file, log_mel_spec_tfm, \
    save_spectrogram, save_mask, log_cqt

NUM_BINS = 12  # Directional binning
NUM_BGS = 3  # Number of background files
DIM_DIVISOR = 32  # To go through UNet, must divide this dim
TASK_DIRECTION = 0  # Different tasks the dataloader can do
TASK_SEPARATION = 1


class SpatialAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, task=0):
        super(SpatialAudioDataset, self).__init__()

        # Data is stored in subdirectories. Get all of them
        self.dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
        self.cache = {}
        self.task = task

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        curr_dir = self.dirs[idx]

        # Get all WAV files in subdirectory
        mixed_audio_files = sorted(
            glob.glob(os.path.join(curr_dir, "*_mixed.wav")))

        # Load mixed data
        mixed_specgrams = []
        for i, mixed_audio_file in enumerate(mixed_audio_files):
            specgram = log_cqt(mixed_audio_file, sample_rate=22500)
            mixed_specgrams.append(torch.from_numpy(specgram))
            #save_spectrogram(specgram, "../data/cqtspectrogram{:02}.png".format(i))
        mixed_data = np.stack(
            mixed_specgrams)  # NUM_MICS x Freq_bins x Time_bins

        if self.task == TASK_DIRECTION:
            return self.direction_labels(mixed_data, curr_dir)

        elif self.task == TASK_SEPARATION:
            return self.unet_voice_labels(mixed_data, curr_dir)

    def direction_labels(self, mixed_data, curr_dir):
        """
        Returns input mixed spectrogram and directional label
        """
        # Now load labels
        with open(os.path.join(curr_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # Get the direction in radians from -pi to pi
        position = metadata["source00"]["position"]  # x,y,z
        angular_direction = np.arctan2(position[1], position[0])
        label = int(
            math.floor((angular_direction + np.pi) * NUM_BINS / (2 * np.pi)))
        return (torch.tensor(mixed_data).float(), torch.tensor(label))

    def unet_voice_labels(self, mixed_data, curr_dir):
        """
        Returns input mixed spectrogram and binary mask for voice class
        """
        # Ground truth voice files
        gt_audio_files = sorted(
            glob.glob(os.path.join(curr_dir, "*_source00_gt.wav")))
        gt_specgrams = []
        for gt_audio_file in gt_audio_files:
            specgram = log_cqt(gt_audio_file, sample_rate=22500)
            gt_specgrams.append(torch.from_numpy(specgram))

        gt_data = torch.stack(gt_specgrams)  # NUM_MICS x Freq_bins x Time_bins

        # Background ground truth specs
        bg_max = np.ones_like(
            gt_data.numpy()) * np.NINF  # NUM_MICS x Freq x Time
        for bg_source_idx in range(1, NUM_BGS + 1):
            gt_bg_files = sorted(
                glob.glob(
                    os.path.join(
                        curr_dir,
                        "*_source{:02d}_gt.wav".format(bg_source_idx))))
            curr_bg_stack = np.zeros_like(bg_max)
            for bg_mic_idx, gt_bg_file in enumerate(gt_bg_files):
                curr_bg_stack[bg_mic_idx, :, :] = log_cqt(gt_bg_file,
                                                          sample_rate=22500)
            bg_max = np.maximum(bg_max, curr_bg_stack)

        mask = gt_data.numpy() > bg_max

        # UNet requires all dims to be divisible by 32
        time_dim = mask.shape[2]
        time_dim_padded = math.ceil(time_dim / DIM_DIVISOR) * DIM_DIVISOR

        # Padded inputs
        input_padded = np.zeros(
            (mixed_data.shape[0], mixed_data.shape[1], time_dim_padded))
        input_padded[:mixed_data.shape[0], :mixed_data.shape[1], :mixed_data.
                     shape[2]] = mixed_data

        # Padded labels
        mask_padded = np.zeros((mask.shape[0], mask.shape[1], time_dim_padded))
        mask_padded[:mask.shape[0], :mask.shape[1], :mask.shape[2]] = mask
        save_mask(mask_padded, "../data/")

        return (torch.tensor(input_padded).float(),
                torch.tensor(mask_padded).float())
