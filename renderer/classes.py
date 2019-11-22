import os
from typing import List

import numpy as np
import torch
import librosa
import soundfile as sf


from constants import SPEED_OF_SOUND, ATTENUATION_ALPHA

INPUT_OUTPUT_TARGET_SAMPLE_RATE = 48000

class Microphone(object):
    def __init__(self, position: List[float]):
        """
        Args:
            position: x,y,z list of floats
        """
        self.position = np.array(position)
        self.buffer = np.array([])
        self.sample_rate = None

    def save(self, filename: str):
        data = self.buffer
        sf.write(filename, data, self.sample_rate)


class SoundSource(object):
    def __init__(self, position: List[float], filename, offset: float = 0.0, duration = None,
                 start_time: float = 0.0):
        assert(len(position) == 3)  # x, y, z
        self.position = np.array(position)

        if type(filename) is str:
            audio, sample_rate = librosa.core.load(filename,  sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE, mono=True,
                                                   offset=offset, duration=duration)

        else:
            audio = np.array([])
            for f in filename:
                curr_audio, sample_rate = librosa.core.load(f,  sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE, mono=True,
                                                            offset=offset, duration=duration)
                audio = np.concatenate((audio, curr_audio))

        self.audio = audio
        self.sample_rate = sample_rate
        self.start_time = start_time


class Scene(object):
    def __init__(self, sources: List[SoundSource], mics: List[Microphone]):
        self.sources = sources
        self.mics = mics

        assert all([x.sample_rate == sources[0].sample_rate for x in sources]), \
            "Sample rate conversion not supported right now"

        self.sample_rate = sources[0].sample_rate

    def render(self, cutoff_time: float, geometric_attenuation = True, atmospheric_attenuation = True):
        """
        Render all sound sources to all microphones.
        Only does ITD.

        cutoff_time: in seconds
        """
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
                if (geometric_attenuation):
                    curr_buffer = np.divide(curr_buffer, distance**2) # attenuation due to energy spreading over area 
                if (atmospheric_attenuation):
                    curr_buffer = np.multiply(curr_buffer, np.exp(-ATTENUATION_ALPHA * distance)) # attenuation due to atmosphere https://en.wikibooks.org/wiki/Engineering_Acoustics/Outdoor_Sound_Propagation
                mic.buffer += np.array(curr_buffer)

    def render_binaural(self, mic_idxs: List[int], output_filename: str, cutoff_time: float):
        """
        Renders a stero output with two chosen mics
        """
        self.render(cutoff_time)
        left_data = self.mics[mic_idxs[0]].buffer
        right_data = self.mics[mic_idxs[1]].buffer
        left_data = np.expand_dims(left_data, 0)
        right_data = np.expand_dims(right_data, 0)

        stereo_data = np.concatenate((left_data, right_data))
        sf.write(output_filename, np.swapaxes(stereo_data, 0, 1), self.sample_rate)


