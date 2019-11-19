import os

import numpy as np

from classes import Microphone, SoundSource, Scene

SOUND_DIR = "../data/input_sounds"
OUTPUT_DIR = "../data/output_sounds"


mic_array = []
radius = 0.1  # Mic array has radius 0.3m
for i in range(8):
    position_x = radius * np.cos(2*np.pi / 8 * i)
    position_y = radius * np.sin(2*np.pi / 8 * i)
    position_z = 0  # Assume planar for now
    mic_array.append(Microphone([position_x, position_y, position_z]))

sound_source_voice = SoundSource([10.0, 0.0, 0.0], os.path.join(SOUND_DIR, "always.flac"))
sound_source_guitar = SoundSource([-10.0, 0.0, 0.0], os.path.join(SOUND_DIR, "guitar.wav"))

scene = Scene([sound_source_voice, sound_source_guitar], mic_array)
scene.render(cutoff_time=6)

for i, mic in enumerate(mic_array):
    output_file = os.path.join(OUTPUT_DIR, "{:05d}.wav".format(i))
    mic.save(output_file)

scene.render_binaural([0, 4], os.path.join(OUTPUT_DIR, "stereo.wav"), cutoff_time=6)