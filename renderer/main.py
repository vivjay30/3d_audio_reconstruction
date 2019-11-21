import os
import json

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

for data_sample_idx in range(10):
    data_dir = os.path.join(OUTPUT_DIR, "{:05d}".format(data_sample_idx))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    random_x1, random_y1 = np.random.uniform(1.0, 15.0, 2)
    random_x2, random_y2 = np.random.uniform(-15.0, 15.0, 2)

    sound_source_voice = SoundSource([random_x1, random_y1, 0.0], os.path.join(SOUND_DIR, "always.flac"))
    sound_source_guitar = SoundSource([random_x2, random_y2, 0.0], os.path.join(SOUND_DIR, "the_accused.wav"))

    scene = Scene([sound_source_voice, sound_source_guitar], mic_array)
    scene.render(cutoff_time=6)

    for i, mic in enumerate(mic_array):
        output_file = os.path.join(data_dir, "mic_{:02d}.wav".format(i))
        mic.save(output_file)

    metadata = {
        "source1" : [random_x1, random_y1, 0.0],
        "source2" : [random_x2, random_y2, 0.0]
    }

    metadata_file = os.path.join(data_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    # scene.render_binaural([0, 4], os.path.join(OUTPUT_DIR, "stereo.wav"), cutoff_time=6)