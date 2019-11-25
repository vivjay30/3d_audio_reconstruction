import os
import json
import random

import numpy as np
import librosa

from d3audiorecon.renderer.classes import Microphone, SoundSource, Scene, \
    INPUT_OUTPUT_TARGET_SAMPLE_RATE

VOICES_DIR = "/projects/grail/vjayaram/d3audiorecon/data/input_data/VCTK-Corpus/wav48"
OUTPUT_DIR = "../data/output_sounds/test_audioset"
BG_DIR = "../data/input_sounds/background"

NUM_MICS = 8
NUM_SCENES = 1000
NUM_FILES = 3  # Voice files are ~2 seconds. Use this many
SCENE_DURATION = 6.0  # For now we are doing scenes of 6 seconds
NUM_BACKGROUNDS = 3  # Number of background sounds present
BG_QUIET = 0.3  # We make all BG sounds a big quieter

# Create mic array first
mic_array = []
radius = 0.3  # Mic array has radius 0.3m
for i in range(NUM_MICS):
    position_x = radius * np.cos(2 * np.pi / NUM_MICS * i)
    position_y = radius * np.sin(2 * np.pi / NUM_MICS * i)
    position_z = 0  # Assume planar for now
    mic_array.append(Microphone([position_x, position_y, position_z]))



all_voices = os.listdir(VOICES_DIR)
all_bg_files = os.listdir(BG_DIR)

# Render a large number of scenes
for data_sample_idx in range(NUM_SCENES):
    metadata = {}
    all_sources = []

    # First generate the voice
    random_voice_dir = os.path.join(VOICES_DIR, random.choice(all_voices))
    voice_files = sorted(os.listdir(random_voice_dir))

    print(data_sample_idx)
    starting_voice_file = random.randint(0, len(voice_files) - NUM_FILES)
    voice_files = voice_files[starting_voice_file:starting_voice_file + NUM_FILES]
    voice_files = [os.path.join(random_voice_dir, x) for x in voice_files]

    # Data dir is 5 digit sequential numerical
    data_dir = os.path.join(OUTPUT_DIR, "{:05d}".format(data_sample_idx))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate random positions for the sources
    random_x, random_y = np.random.uniform(-10.0, 10.0, 2)
    sound_source_voice = SoundSource([random_x, random_y, 0.0], voice_files)
    all_sources.append(sound_source_voice)
    metadata["source00"] = {
        "position" : [random_x, random_y, 0.0],
        "filename" : os.path.join(data_dir, "gt_voice.wav")
    }
    

    # Now generate background
    for bg_source_idx in range(NUM_BACKGROUNDS):
        bg_file = random.choice(all_bg_files)

        # We start by using this music file
        bg_data, bg_sr = librosa.core.load(
            os.path.join(BG_DIR, bg_file),
            sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
            mono=True,
        )
        bg_data *= BG_QUIET  # Quiet the background

        random_x, random_y = np.random.uniform(-10.0, 10.0, 2)
        sound_source_bg = SoundSource(
            [random_x, random_y, 0.0],
            data=bg_data,
            sr=bg_sr)
        all_sources.append(sound_source_bg)
        metadata["source{:02d}".format(bg_source_idx + 1)] = {
            "position" : [random_x, random_y, 0.0],
            "filename" : os.path.join(BG_DIR, bg_file)
        }

    scene = Scene(all_sources, mic_array)
    scene.render(cutoff_time=SCENE_DURATION)

    # Write every mic buffer to outputs
    for i, mic in enumerate(mic_array):
        output_prefix = os.path.join(data_dir, "mic{:02d}_".format(i))
        mic.save(output_prefix)
        mic.reset()

    metadata_file = os.path.join(data_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # scene.render_binaural([0, 4], os.path.join(OUTPUT_DIR, "stereo.wav"), cutoff_time=6)
