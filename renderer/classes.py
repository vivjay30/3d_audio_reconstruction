import os
from typing import List

import numpy as np
import torch
import torchaudio

from constants import SPEED_OF_SOUND


class Microphone(object):
    def __init__(self, position: List[float]):
        self.position = np.array(position)
        self.buffer = np.array([])
        self.sample_rate = None

    def save(self, filename: str):
        data = torch.Tensor(self.buffer).view(1, -1)
        torchaudio.save(filename, data, self.sample_rate)


class SoundSource(object):
    def __init__(self, position: List[float], filename: str, start_time: float = 0.0):
        assert(len(position) == 3)  # x, y, z
        self.position = np.array(position)
        audio, sample_rate = torchaudio.load(filename)
        audio = np.mean(audio.numpy(), axis=0)
        self.audio = audio
        self.sample_rate = sample_rate
        self.start_time = start_time



class Scene(object):
    def __init__(self, sources: List[SoundSource], mics: List[Microphone]):
        self.sources = sources
        self.mics = mics

        print([x.sample_rate == sources[0].sample_rate for x in sources])
        assert all([x.sample_rate == sources[0].sample_rate for x in sources]), \
            "Sample rate conversion not supported right now"

        self.sample_rate = sources[0].sample_rate

    def render(self, cutoff_time: float):
        for mic in self.mics:
            # Initialize mic buffer to be empty correct size
            total_samples = int(cutoff_time * self.sample_rate)
            mic.buffer = np.zeros((total_samples))
            mic.sample_rate = self.sample_rate

            # Render all sources to that mic
            for source in self.sources:
                distance = np.linalg.norm(mic.position - source.position)
                curr_start_time = source.start_time + distance / SPEED_OF_SOUND
                curr_start_samples = int(source.sample_rate * curr_start_time)
                curr_buffer = np.concatenate((np.zeros((curr_start_samples)).astype(np.float64),
                                              source.audio))
                curr_buffer = curr_buffer[:total_samples]
                mic.buffer += np.array(curr_buffer)

    def render_binaural(self, mic_idxs: List[int], output_filename: str, cutoff_time: float):
        self.render(cutoff_time)
        left_data = self.mics[mic_idxs[0]].buffer
        right_data = self.mics[mic_idxs[1]].buffer
        left_data = np.expand_dims(left_data, 0)
        right_data = np.expand_dims(right_data, 0)

        stereo_data = np.concatenate((left_data, right_data))
        torchaudio.save(output_filename, torch.tensor(stereo_data), self.sample_rate)






