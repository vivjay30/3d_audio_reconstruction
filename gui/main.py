"""
Install dependencies: python3 -m pip install -U pg --user

Small applet for bulding audio scenes

Be able to instantiate and place speakers for rendering SoundScenes(tm)

Current behavior:
Right click to add/delete sound source. Left click to select sound source and move it around.

The app always has a single foregorund source on the screen. Left select to click and drag
this source to move it around. Click the "New Background Source" button to instantiate a new
background source. User may add up to NUM_BG_SOURCES number of a background sources.

When a source is selected, it will turn red. Two options will be possible: replace source and
delete. Foreground source cannot be deleted.

Right panel has button to render sound setup on right. Once rendered specgram is shown displayed.

"""
import os
import sys
import pygame as pg
import math
import random
import glob
import subprocess
import time

# import Button class
# Button class from https://github.com/Mekire/pg-button
from button import Button

import numpy as np
import librosa
import shutil
import json

from d3audiorecon.renderer.classes import Microphone, SoundSource, Scene, \
    INPUT_OUTPUT_TARGET_SAMPLE_RATE

from d3audiorecon.tools.utils import read_file, log_mel_spec_tfm, \
    save_spectrogram, save_mask, log_cqt

DISPLAY_HEIGHT = 800 # px
SOURCE_DISPLAY_WIDTH = DISPLAY_HEIGHT # px, UI area to position sources, room assumed to be square
SPEC_DISPLAY_WIDTH = 400 # px, right side of screen will be used to show specgrams
SPECGRAM_SIZE = (368, 128)

ROOM_SIZE = 10 # meters # assumes square room, room center is (ROOM_SIZE/2, ROOM_SIZE/2)

# Represents conversion between display pixels and meters
PIXELS_PER_METER = DISPLAY_HEIGHT // ROOM_SIZE

# Size of bin in degrees
BIN_SIZE = 30

# Color defnititions
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (235,51,35)
DARK_RED =(235*0.7,51*0.7,35*0.7)
BLUE = (0,71,113)
LIGHT_BLUE = ((0*1.3,71*1.3,113*1.3))
LIGHTER_BLUE = ((0*1.8,71*1.8,113*1.8))
GREEN = (0,255,0)
BLACK = (0,0,0)
ORANGE = (255,180,0)
LIGHT_GREY = (235, 235, 235)
LIGHT_GREY_ALPHA = (200, 200, 243, 230)
HOVER_LIGHT_GREY = (220, 220, 220)
DARK_GREY = (37, 37, 38)
HOVER_DARK_GREY = (37*1.8, 37*1.8, 38*1.8)

# Dictionary of different source types
SOURCE_TYPES = {0:"foreground", 1:"background"}
BG_PATH = "../data/input_sounds/background/"
FG_PATH = "../data/input_sounds/voices/"
OUTPUT_DIR = "/Users/vivekjayaram/Documents/CSProjects/grail_mounted/d3audiorecon/gui/"

# Delete old files
subprocess.call(["./delete_remote.sh"])
output_data_dir = os.path.dirname(os.path.abspath(__file__))
output_data_dir = output_data_dir + "/output/"
bin_in = None

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)

# Maximum number of each type of sources
NUM_FG_SOURCES = 1
NUM_BG_SOURCES = 3

# Number of milliseconds to display non-null status for
STATUS_HOLD_TIME = 2000

# Init
pg.init()

# Define base font
font = pg.font.SysFont('helvetica', 14)
font_large = pg.font.SysFont('helvetica', 24)

# Define default button style
BUTTON_STYLE = {"font" : font,
                "hover_color" : HOVER_DARK_GREY,
                "clicked_color" : BLUE,
                "clicked_font_color" : WHITE,
                "hover_font_color" : WHITE}

# Define red button style
RED_BUTTON_STYLE = {"font" : font,
                "hover_color" : RED,
                "clicked_color" : ORANGE,
                "clicked_font_color" : WHITE,
                "hover_font_color" : WHITE}

# Define blue button style
BLUE_BUTTON_STYLE = {"font" : font_large,
                "hover_color" : LIGHT_BLUE,
                "clicked_color" : LIGHTER_BLUE,
                "clicked_font_color" : WHITE,
                "hover_font_color" : WHITE}

# Helper functions
# Translates between meter coodinates and pixel coordinates
def pixel_to_meter_pos(position_in_pixels: tuple):
    return (((position_in_pixels[0]- SOURCE_DISPLAY_WIDTH//2) / PIXELS_PER_METER, (DISPLAY_HEIGHT//2 - position_in_pixels[1]) / PIXELS_PER_METER))

def meter_to_pixel_pos(position_in_meters: tuple):
    return (position_in_meters[0] * PIXELS_PER_METER + SOURCE_DISPLAY_WIDTH//2, -1*position_in_meters[1] *  PIXELS_PER_METER + DISPLAY_HEIGHT//2)

# Grab a random rile from a specific directory
def pick_random_file(path: str):
    all_wav_files = glob.glob(os.path.join(path, "*.wav"))
    return random.choice(all_wav_files)

# Takes bin_in from angle network computes angle range corresponding to that bin
# 0th bin_in is the bin going counterclockwise starting at the ray from the origin to (-1, 0)
# 0th bin_out is the bin going counterclockwise starting at the ray from the origin to (1, 0)
def bin_index_to_angle_range(bin_idx: int):
    num_bins = 360 // BIN_SIZE
    # peform 180 deg rotation of bins
    bin_out = (bin_idx + num_bins/2) % num_bins
    # return tuple with angle range for bin boundaries
    return (math.radians((bin_out * BIN_SIZE) % 360), math.radians(((bin_out+1) * BIN_SIZE) % 360))

# Constructor for sources. Each source carries these attributes
class Source(pg.sprite.Sprite):
    def __init__(self, xpos, ypos, idx, source_type):
        super(Source, self).__init__()
        # Possible image representations for sources
        self.images = [pg.image.load("images/foreground.png").convert_alpha(),\
                        pg.image.load("images/foreground_selected.png").convert_alpha(),\
                        pg.image.load("images/background.png").convert_alpha(),\
                        pg.image.load("images/background_selected.png").convert_alpha()]
        self.source_type = source_type # 0: foreground, 1: background
        if self.source_type == 0: # If source is a foreground type
            self.image = self.images[0]
        if self.source_type == 1: # If source is a background type
            self.image = self.images[1]
        self.selected = False # triggered whenever a quick click on sprite results in selection
        self.clicked = False # triggered whenever the mouse button is down on the sprite
        self.rect = self.image.get_rect()
        self.rect.x = xpos - self.rect.width//2
        self.rect.y = ypos - self.rect.height//2
        self.idx = idx
        # Position in meters with (0,0) as center of room
        self.position = pixel_to_meter_pos(self.rect.center)
        # Angle in degrees
        self.angle = math.degrees(math.atan2(self.position[1],self.position[0]))
        # Label associated with source instance
        self.text = SOURCE_TYPES[self.source_type] + " source: "+ ('%.4f' % self.position[0]) + ", " + ('%.4f' % self.position[1]) + \
                            "   Angle: " + ('%.4f' % self.angle)
        # Sound source
        if self.source_type == 0: # If source is a foreground type
            self.sound_path = pick_random_file(FG_PATH)
        if self.source_type == 1: # If source is a background type
            self.sound_path = pick_random_file(BG_PATH)
        self.sound = pg.mixer.Sound(self.sound_path)

# Intialize, render screen, start clock
screen = pg.display.set_mode((SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH, DISPLAY_HEIGHT))
pg.display.set_caption("Speaker renderer")

# Define separate surface for the source UI panel that can handle per-pixel transparencies
# Used only for transparent polygons
source_ui_area = pg.Surface((SOURCE_DISPLAY_WIDTH, DISPLAY_HEIGHT), pg.SRCALPHA)

# Start pygame clock
clock = pg.time.Clock()

# Exit flag
exit = False

# Initialize a group of sprites
source_list = []# pg.sprite.Group()
clicked_on_nothing = False # If user clicks on nothing flag
pos_offset = (-1,-1) # Holds position of mouse pos to sprite pos offset for drag and drop
clicked_on_a_button = False # If user clicks on a button flag

# List of different potential status messages
statuses = ["", \
        "Max number of background sources: " + str(NUM_BG_SOURCES), \
        "Foreground source cannot be deleted", \
        "No sources selected", \
        "Source sound changed"]
# Current status to be displayed
current_status = statuses[0]
status_timer = pg.time.get_ticks()

# Dummy function
def dummy_function():
    print("hi")

# Makes a new Source and adds it to the source list
def add_source(x=300, y=300, idx=len(source_list)+1, source_type=1):
    global source_list
    global Source
    # Ensure that max number of sources is not exceeded
    if len(source_list) < (NUM_BG_SOURCES + NUM_FG_SOURCES):
        source_list.append(Source(min(x,SOURCE_DISPLAY_WIDTH), y, idx, source_type))
    # Set status to max number exceeded warning
    else:
        global current_status
        current_status = statuses[1]
        global status_timer
        status_timer = pg.time.get_ticks()

# Grabs a random sound from corresponding directory and assign it to the source
def change_source(source):
        pg.mixer.fadeout(300)
        if source.source_type == 0: # If source is a foreground type
            source.sound_path = pick_random_file(FG_PATH)
            source.sound = pg.mixer.Sound(source.sound_path)
        if source.source_type == 1: # If source is a background type
            source.sound_path = pick_random_file(BG_PATH)
            source.sound = pg.mixer.Sound(source.sound_path)
        global current_status
        current_status = statuses[4]
        global status_timer
        status_timer = pg.time.get_ticks()


# Iterate through all selected sources and delete those that are selected
def change_source_selected():
    global source_list
    for source in source_list:
        if source.selected:
            change_source(source)
            return
    global current_status
    current_status = statuses[3]
    global status_timer
    status_timer = pg.time.get_ticks()

# Plays sound associated with source
def play_source(source):
    pg.mixer.fadeout(300)
    source.sound.play()

# Iterate through all selected sources and delete those that are selected
def play_source_selected():
    global source_list
    for source in source_list:
        if source.selected:
            play_source(source)
            return
    global current_status
    current_status = statuses[3]
    global status_timer
    status_timer = pg.time.get_ticks()

# Remove source
def delete_source(source):
    pg.mixer.fadeout(300)
    global source_list
    # Remove source from source list as long as it is not a foreground source
    if not source.source_type == 0:
        source_list.remove(source)
    # Otherwise, let user know they cannot delete foreground source
    elif source.source_type == 0:
        global current_status
        current_status = statuses[2]
        global status_timer
        status_timer = pg.time.get_ticks()


# Iterate through all selected sources and delete those that are selected
def delete_source_selected():
    global source_list
    for source in source_list:
        if source.selected:
            delete_source(source)
            return
    global current_status
    current_status = statuses[3]
    global status_timer
    status_timer = pg.time.get_ticks()

global specgram_image
specgram_image = None
global rendered 
rendered = False
global separated_image
separated_image = None
global signal_separated_done
signal_separated_done = False
global loaded_output_dir
loaded_output_dir = False
global loaded_output_separated
loaded_output_separated = False

def render():
    subprocess.call(["./delete_remote.sh"])
    global loaded_output_dir
    loaded_output_dir = False
    global loaded_output_separated
    loaded_output_separated = False
    global signal_separated_done
    signal_separated_done = False
    global specgram_image
    specgram_image = None
    global separated_image
    separated_image = None
    global bin_in
    bin_in = None
    global rendered
    rendered = False
    NUM_MICS = 8
    RENDER_TIME = 6.0
    all_sources = []
    mic_array = []
    metadata = {}

    radius = 0.3  # Mic array has radius 0.3m
    for i in range(NUM_MICS):
        position_x = radius * np.cos(2 * np.pi / 8 * i)
        position_y = radius * np.sin(2 * np.pi / 8 * i)
        position_z = 0  # Assume planar for now
        mic_array.append(Microphone([position_x, position_y, position_z]))

    # generate the sound sources
    for i in range(0, len(source_list)):
        print(source_list[i].position[0])
        print(source_list[i].position[1])
        print(source_list[i].sound_path)

        reduce_factor = 1.0 if i == 0 else 0.5
        all_sources.append(SoundSource([source_list[i].position[0], source_list[i].position[1], 0.0],
                                       filename=source_list[i].sound_path, reduce_factor=reduce_factor))
        metadata["source{:02d}".format(i)] = {
            "position" : [source_list[i].position[0], source_list[i].position[1], 0.0],
            "filename" : source_list[i].sound_path
        }

    scene = Scene(all_sources, mic_array)
    scene.render(cutoff_time=RENDER_TIME, volume_boost=4.0)


    output_data_dir = os.path.dirname(os.path.abspath(__file__))
    output_data_dir = output_data_dir + "/output/"
    
    if os.path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)

    os.makedirs(output_data_dir)

    # Write every mic buffer to outputs
    for i, mic in enumerate(mic_array):
        output_prefix = os.path.join(output_data_dir, "mic{:02d}_".format(i))
        mic.save(output_prefix)
        mic.reset()

    metadata_file = os.path.join(output_data_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # spectrogram
    mixed_audio_file = output_data_dir + "mic00_mixed.wav"
    specgram = log_cqt(mixed_audio_file, sample_rate=22500)
    save_spectrogram(specgram, output_data_dir + "spectrogram.png")

    specgram_image = pg.image.load(output_data_dir + "spectrogram.png")
    specgram_image = pg.transform.flip(specgram_image, False, True)
    specgram_image = pg.transform.scale(specgram_image, SPECGRAM_SIZE)

    processes = []
    for i in range(8):
        p = subprocess.Popen(["scp", "output/mic{:02d}_mixed.wav".format(i), "vjayaram@lungo.cs.washington.edu:/projects/grail/vjayaram/d3audiorecon/gui/output/"])
        processes.append(p)

    for p in processes:
        p.wait()

    subprocess.Popen(["./forward.sh"])

    rendered = True
    return source_list

def play_render():
    rendered_sound_dir = os.path.dirname(os.path.abspath(__file__)) + "/output/mic00_mixed.wav"
    pg.mixer.Sound(rendered_sound_dir).play()
    return

def play_separated():
    separated_sound_dir = "unet_cqt_output.wav"
    pg.mixer.Sound(separated_sound_dir).play()
    return

# Button definitions and positioning
new_source_button = Button((0,0,200,24), DARK_GREY, add_source, text="New background source", **BUTTON_STYLE)
new_source_button.rect.bottomleft = (screen.get_rect().bottomleft[0] + 5, screen.get_rect().bottomleft[1] - 5)

change_source_button = Button((0,0,130,24), DARK_GREY, change_source_selected, text="Change source", **BUTTON_STYLE)
change_source_button.rect.bottomleft = (new_source_button.rect.bottomright[0] + 5, screen.get_rect().bottomleft[1] - 5)

play_source_button = Button((0,0,130,24), DARK_GREY, play_source_selected, text="Play source", **BUTTON_STYLE)
play_source_button.rect.bottomleft = (change_source_button.rect.bottomright[0] + 5, screen.get_rect().bottomleft[1] - 5)

delete_source_button = Button((0,0,120,24), DARK_RED, delete_source_selected, text="Delete source", **RED_BUTTON_STYLE)
delete_source_button.rect.bottomleft = (play_source_button.rect.bottomright[0] + 5, screen.get_rect().bottomleft[1] - 5)

render_button = Button((0,0,368,64), BLUE, render, text="RENDER AND RUN", **BLUE_BUTTON_STYLE)
render_button.rect.center = (SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2, 48)

# Define area for rendered specgram placeholder
specgram_rect  = pg.Rect((SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2 - SPECGRAM_SIZE[0]//2, render_button.rect.bottom + 16), SPECGRAM_SIZE)

play_rendered_button = Button((0,0,368,24), DARK_GREY, play_render, text="Play rendered mix", **BUTTON_STYLE)
play_rendered_button.rect.center = (SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2, specgram_rect.bottom + play_rendered_button.rect.height//2 + 16)

separated_rect  = pg.Rect((SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2 - SPECGRAM_SIZE[0]//2, play_rendered_button.rect.bottom + 16), SPECGRAM_SIZE)

play_separated_button = Button((0,0,368,24), DARK_GREY, play_separated, text="Play separated source mix", **BUTTON_STYLE)
play_separated_button.rect.center = (SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2, separated_rect.bottom + play_separated_button.rect.height//2 + 16)


button_list = [new_source_button, change_source_button, play_source_button, delete_source_button, render_button, play_rendered_button, play_separated_button]

# Add one foreground source
add_source(x=random.randint(0.25*SOURCE_DISPLAY_WIDTH, 0.75*SOURCE_DISPLAY_WIDTH), y=random.randint(0.25*DISPLAY_HEIGHT, 0.75*DISPLAY_HEIGHT), source_type=0)


lastTimestamp = time.time()
# Main program loop
while not exit:
    # Get pg  event
    for event in pg.event.get():
        # Quit if pg is quit
        if event.type == pg.QUIT:
            exit = True

        # status timer handler
        if (pg.time.get_ticks() - status_timer) > STATUS_HOLD_TIME:
            current_status = statuses[0]

        # Button interaction logic
        new_source_button.check_event(event)
        change_source_button.check_event(event)
        play_source_button.check_event(event)
        delete_source_button.check_event(event)
        render_button.check_event(event)
        play_rendered_button.check_event(event)
        play_separated_button.check_event(event)

        # Set clicked on a button flag
        for button in button_list:
            if button.clicked:
                clicked_on_a_button = True
                break
            else:
                clicked_on_a_button = False

        # Source interaction logic
        # If the user didn't click on a button
        if not clicked_on_a_button:
            # If any mouse button is  pressed
            if event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                x, y = pos[0], pos[1]

                # For right click
                if event.button == 3:
                    something_there_already = False
                    # For all sprites
                    for source in source_list:
                        # If clicked, and something is there,  delete everything that's right-clicked
                        if source.rect.collidepoint(pos):
                            something_there_already = True
                            delete_source(source)

                    #  If there's nothing there, instantiate a new sprite
                    if not something_there_already:
                        add_source(x=x, y=y,idx=len(source_list)+1, source_type=1) # TODO: fix iteration scheme

                # If left-clicked
                elif event.button == 1:
                    for source in source_list:
                        source.selected = False
                    # For the first source in the stack that's clicked, mark that it's clicked and change it's status to selected
                    for source in source_list:
                        if source.rect.collidepoint(pos):
                            # if not source.clicked:
                            #     source.selected = not source.selected
                            source.selected = True
                            source.clicked = True
                            clicked_on_nothing = False
                            if pos_offset == (-1,-1):
                                pos_offset = (x - source.rect.topleft[0], y - source.rect.topleft[1])
                            break # if there are overlapping sprites only select one
                        clicked_on_nothing = True # after checking that all items are not clicked

                    if clicked_on_nothing == True: # if user has left clicked in open space
                        for source in source_list:
                            source.selected = False # de-select currently selected sprite

            # Mark all sources as unclicked if no mouse button is pressed
            if event.type == pg.MOUSEBUTTONUP:
                for source in source_list:
                    source.clicked = False
                pos_offset = (-1,-1)

            # Update the clicked sprite's position for drag behavior
            for source in source_list:
                if source.clicked == True:
                    bin_in = None
                    pos = pg.mouse.get_pos()
                    source.rect.x = min(pos[0] - pos_offset[0], SOURCE_DISPLAY_WIDTH-(source.rect.width//2))
                    source.rect.y = pos[1] - pos_offset[1]
                    source.position = pixel_to_meter_pos(source.rect.center)
                    source.angle = math.degrees(math.atan2(source.position[1],source.position[0]))
                    source.text = SOURCE_TYPES[source.source_type] + " source: "+ ('%.4f' % source.position[0]) + ", " + ('%.4f' % source.position[1]) + \
                            "   Angle: " + ('%.4f' % source.angle)
                # Update source's icon if selected
                if source.selected:
                    if source.source_type == 0:
                        source.image = source.images[1]
                    elif source.source_type == 1:
                        source.image = source.images[3]
                else:
                    if source.source_type == 0:
                        source.image = source.images[0]
                    elif source.source_type == 1:
                        source.image = source.images[2]


    # Clear screen
    screen.fill(WHITE)
    source_ui_area.fill((0, 0, 0, 0))

    # Draw center indicator
    mic_img = pg.image.load("images/mic.png").convert_alpha()
    mic_img_rect = mic_img.get_rect()
    screen.blit(mic_img, (SOURCE_DISPLAY_WIDTH//2 - mic_img_rect.width//2, SOURCE_DISPLAY_WIDTH//2 - mic_img_rect.height//2))

    # Draw grid
    for i in range(-ROOM_SIZE//2, ROOM_SIZE//2):
        pg.draw.line(screen, DARK_GREY, meter_to_pixel_pos((i, -ROOM_SIZE//2)), meter_to_pixel_pos((i, ROOM_SIZE//2)), 1)
    for i in range(-ROOM_SIZE//2, ROOM_SIZE//2):
        pg.draw.line(screen, DARK_GREY, meter_to_pixel_pos((-ROOM_SIZE//2, i)), meter_to_pixel_pos((ROOM_SIZE//2, i)), 1)

    if rendered:
        if (not loaded_output_separated) and os.path.exists("unet_cqt_output.wav"):
            time.sleep(2)
            signal_separated_done = True
            loaded_output_separated = True

            separated_audio_file = "unet_cqt_output.wav"
            specgram = log_cqt(separated_audio_file, sample_rate=22500)
            save_spectrogram(specgram, os.path.join("output/", "separated_spectrogram.png"))

            separated_image = pg.image.load(os.path.join("output/", "separated_spectrogram.png"))
            separated_image = pg.transform.flip(separated_image, False, True)
            separated_image = pg.transform.scale(separated_image, SPECGRAM_SIZE)
            print(separated_image)

        if (not loaded_output_dir) and os.path.exists("output.json"):
            time.sleep(2)
            loaded_output_dir = True

            with open("output.json") as f:
                data = json.load(f)
                bin_in = data["bin"]

        # If the bin_in from angle network is defined, draw the corresponding shaded polygon
        if bin_in is not None:
            angle_range = bin_index_to_angle_range(bin_in)


            a = int(2*ROOM_SIZE*math.cos(angle_range[0]))
            b = int(2*ROOM_SIZE*math.sin(angle_range[0]))
            c = int(2*ROOM_SIZE*math.cos(angle_range[1]))
            d = int(2*ROOM_SIZE*math.sin(angle_range[1]))
            pg.draw.polygon(source_ui_area, LIGHT_GREY_ALPHA, \
                            [meter_to_pixel_pos((0,0)), \
                             meter_to_pixel_pos((a,b)), \
                             meter_to_pixel_pos((c,d))])
        screen.blit(source_ui_area, source_ui_area.get_rect())

        lastTimestamp = time.time()
    

    # Draw specgram placeholder area
    pg.draw.rect(screen, LIGHT_GREY, specgram_rect)

    # TODO if specgram exists, draw it in the specgram area
    if specgram_image is not None:
        screen.blit(specgram_image, (SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2 - SPECGRAM_SIZE[0]//2, render_button.rect.bottom + 16))

    # Draw separated placeholder area
    pg.draw.rect(screen, LIGHT_GREY, separated_rect)

    # TODO if separated exists, draw it in the separated area
    if separated_image is not None:
        screen.blit(separated_image, (SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH//2 - SPECGRAM_SIZE[0]//2, play_rendered_button.rect.bottom + 16))


    # TODO repeat for source separation network too

    # Draw line separating source UI panel area from specgram panel
    pg.draw.line(screen, DARK_GREY, (SOURCE_DISPLAY_WIDTH, 0), (SOURCE_DISPLAY_WIDTH, DISPLAY_HEIGHT), 5)

    # Update button looks
    new_source_button.update(screen)
    change_source_button.update(screen)
    play_source_button.update(screen)
    delete_source_button.update(screen)
    render_button.update(screen)
    if rendered:
        play_rendered_button.update(screen)
    if signal_separated_done:
        play_separated_button.update(screen)

    # Blit status text box
    text = font.render(current_status, True, RED, WHITE)
    screen.blit(text, (new_source_button.rect.x, new_source_button.rect.y - text.get_rect().height - 3))

    # Draw sources and associated label to the screen
    i = 0
    for source in source_list:
        screen.blit(source.image, (source.rect.x,source.rect.y))
        text = font.render(source.text, True, DARK_GREY, WHITE)
        screen.blit(text, (3,3+i*14))
        i += 1

    # Update the screen
    pg.display.update()

    # 60 fps
    clock.tick(60)

# Quit cleanly if game is quit
pg.quit()


