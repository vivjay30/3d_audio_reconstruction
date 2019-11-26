import argparse
import os
import json
import random
import multiprocessing.dummy as mp

import numpy as np
import librosa

from d3audiorecon.renderer.classes import Microphone, SoundSource, Scene, \
    INPUT_OUTPUT_TARGET_SAMPLE_RATE

POOL_SIZE = 25

def generate_mic_array(args):
    """
    Generate a circular mic array with a fixed radius
    """
    mic_array = []
    radius = 0.3  # Mic array has radius 0.3m
    for i in range(args.num_mics):
        position_x = radius * np.cos(2 * np.pi / args.num_mics * i)
        position_y = radius * np.sin(2 * np.pi / args.num_mics * i)
        position_z = 0  # Assume planar for now
        mic_array.append(Microphone([position_x, position_y, position_z]))
    return mic_array


def verify_args(args):
    """
    Check that the input directories are valid
    """
    # Voices dir
    all_voices = os.listdir(args.voices_dir)
    if len(all_voices) == 0:
        raise ValueError("No directories found in {}".format(args.voices_dir))
    args.all_voices = all_voices

    # BG dir
    all_bg_files = sorted(os.listdir(args.bg_sounds_dir))
    if len(all_bg_files) == 0:
        raise ValueError("No files found in {}".format(args.bg_sounds_dir))
    args.all_bg_files = all_bg_files


def main(args):
    verify_args(args)

    # Render a large number of scenes
    def generate_sample(data_sample_idx):
        mic_array = generate_mic_array(args)
        print(data_sample_idx)
        metadata = {}
        all_sources = []

        # First generate the voice
        random_voice_dir = os.path.join(args.voices_dir, random.choice(args.all_voices))
        voice_files = sorted(os.listdir(random_voice_dir))

        starting_voice_file = random.randint(0, len(voice_files) - args.num_voices_concat)
        voice_files = voice_files[starting_voice_file:starting_voice_file + args.num_voices_concat]
        voice_files = [os.path.join(random_voice_dir, x) for x in voice_files]

        # Data dir is 5 digit sequential numerical
        output_data_dir = os.path.join(args.output_dir, "{:05d}".format(data_sample_idx))
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        # Generate random positions for the voice
        random_x, random_y = np.random.uniform(-5.0, 5.0, 2)
        sound_source_voice = SoundSource([random_x, random_y, 0.0], voice_files)
        all_sources.append(sound_source_voice)
        metadata["source00"] = {
            "position" : [random_x, random_y, 0.0],
            "filename" : os.path.join(output_data_dir, "gt_voice.wav")
        }
        

        # Generate a number of background sources
        for bg_source_idx in range(args.num_backgrounds):
            bg_file = random.choice(args.all_bg_files)

            # We start by using this music file
            bg_data, bg_sr = librosa.core.load(
                os.path.join(args.bg_sounds_dir, bg_file),
                sr=INPUT_OUTPUT_TARGET_SAMPLE_RATE,
                mono=True,
            )
            bg_data *= args.bg_reduce_factor  # Quiet the background

            random_x, random_y = np.random.uniform(-10.0, 10.0, 2)
            sound_source_bg = SoundSource(
                [random_x, random_y, 0.0],
                data=bg_data,
                sr=bg_sr)
            all_sources.append(sound_source_bg)
            metadata["source{:02d}".format(bg_source_idx + 1)] = {
                "position" : [random_x, random_y, 0.0],
                "filename" : os.path.join(args.bg_sounds_dir, bg_file)
            }

        scene = Scene(all_sources, mic_array)
        scene.render(cutoff_time=args.scene_duration)

        # Write every mic buffer to outputs
        for i, mic in enumerate(mic_array):
            output_prefix = os.path.join(output_data_dir, "mic{:02d}_".format(i))
            mic.save(output_prefix)
            mic.reset()

        metadata_file = os.path.join(output_data_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        # scene.render_binaural([0, 4], os.path.join(OUTPUT_DIR, "stereo.wav"), cutoff_time=6)

    # Multi-threading
    pool = mp.Pool(POOL_SIZE)
    pool.map(generate_sample, range(args.num_scenes))
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render sounds to a mic array')
    parser.add_argument("voices_dir", type=str, help="Path to voices dir from VCTK dataset")
    parser.add_argument("bg_sounds_dir", type=str, help="Path to background sounds (non voices")
    parser.add_argument("output_dir", type=str, help="Path to output results")
    parser.add_argument("--num-scenes", type=int, default=1000, help="Number of scenes to render")
    parser.add_argument("--num-mics", type=int, default=8, help="Number of mics, default config is a circle")
    parser.add_argument("--scene-duration", type=float, default=6.0, help="All output scenes are this length")
    parser.add_argument("--num-backgrounds", type=int, default=3, help="Number of background sounds per scene")
    parser.add_argument("--bg-reduce-factor", type=float, default=0.5, help="Reduce the volume of the background")
    parser.add_argument("--num-voices-concat", type=int, default=3, help="Number of voice files to concatenate as foreground")

    main(parser.parse_args())

