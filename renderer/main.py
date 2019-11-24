import os
import json
import random

import numpy as np
import librosa

from d3audiorecon.renderer.classes import Microphone, SoundSource, Scene, \
    INPUT_OUTPUT_TARGET_SAMPLE_RATE

SOUND_DIR = "../data/input_sounds"
OUTPUT_DIR = "../data/output_sounds/test_quiet_attenuated"

NUM_MICS = 8
NUM_SCENES = 1000
VOICE_FILES = 3  # Voice files are ~2 seconds. Use this many
SCENE_DURATION = 6.0  # For now we are doing scenes of 6 seconds

# Create mic array first
mic_array = []
radius = 0.3  # Mic array has radius 0.3m
for i in range(NUM_MICS):
    position_x = radius * np.cos(2 * np.pi / NUM_MICS * i)
    position_y = radius * np.sin(2 * np.pi / NUM_MICS * i)
    position_z = 0  # Assume planar for now
    mic_array.append(Microphone([position_x, position_y, position_z]))

# We start by using this music file
music_data, music_sr = librosa.core.load(
    os.path.join(SOUND_DIR, "music/2436.wav"),
    sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
    mono=True,
)
music_data *= 0.3  # Quiet the music

# Render a large number of scenes
for data_sample_idx in range(NUM_SCENES):
    print(data_sample_idx)
    starting_voice_idx = random.randint(1, 100 - VOICE_FILES)  # We have 100 files

    # Data dir is 5 digit sequential numerical
    data_dir = os.path.join(OUTPUT_DIR, "{:05d}".format(data_sample_idx))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate random positions for the sources
    random_x0, random_y0 = np.random.uniform(-10.0, 10.0, 2)
    random_x1, random_y1 = np.random.uniform(-10.0, 10.0, 2)


    voice_files = [
        os.path.join(SOUND_DIR,
                     "voices/p239_{:03d}.wav".format(starting_voice_idx + x))
        for x in range(VOICE_FILES)
    ]
    sound_source_voice = SoundSource([random_x0, random_y0, 0.0], voice_files)
    
    # Random select a 6 second clip from our piano song, render scene
    music_start_point = random.randint(
        0, int(len(music_data) - SCENE_DURATION * music_sr))
    sound_source_music = SoundSource(
        [random_x1, random_y1, 0.0],
        data=music_data[music_start_point:int(music_start_point +
                                              SCENE_DURATION * music_sr)],
        sr=music_sr)

    scene = Scene([sound_source_voice, sound_source_music], mic_array)
    scene.render(cutoff_time=SCENE_DURATION)

    # Write every mic buffer to outputs
    for i, mic in enumerate(mic_array):
        output_prefix = os.path.join(data_dir, "mic{:02d}_".format(i))
        mic.save(output_prefix)
        mic.reset()

    # Save ground truths
    sound_source_music.save(os.path.join(data_dir, "gt_bg.wav"))
    sound_source_voice.save(os.path.join(data_dir, "gt_voice.wav"))

    metadata = {
        "source00": {
            "position": [random_x0, random_y0, 0.0],
            "filename": os.path.join(data_dir, "gt_voice.wav"),
        },
        "source01": {
            "position": [random_x1, random_y1, 0.0],
            "filename": os.path.join(data_dir, "gt_bg.wav")
        }
    }

    metadata_file = os.path.join(data_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    # scene.render_binaural([0, 4], os.path.join(OUTPUT_DIR, "stereo.wav"), cutoff_time=6)
