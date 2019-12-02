""" 
Install dependencies: python3 -m pip install -U pygame --user

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
import pygame
import math
from random import randint

# import Button class
# Button class from https://github.com/Mekire/pygame-button
from button import Button

DISPLAY_HEIGHT = 800 # px
SOURCE_DISPLAY_WIDTH = DISPLAY_HEIGHT # px, UI area to position sources, room assumed to be square
SPEC_DISPLAY_WIDTH = 400 # px, right side of screen will be used to show specgrams

ROOM_SIZE = 20 # meters

# Represents conversion between display pixels and meters
PIXELS_PER_METER = DISPLAY_HEIGHT // ROOM_SIZE

# Color defnititions
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (235,51,35)
DARK_RED =(235*0.7,51*0.7,35*0.7)
BLUE = (0,71,113)
GREEN = (0,255,0)
BLACK = (0,0,0)
ORANGE = (255,180,0)
LIGHT_GREY = (243, 243, 243)
HOVER_LIGHT_GREY = (220, 220, 220)
DARK_GREY = (37, 37, 38)
HOVER_DARK_GREY = (37*1.8, 37*1.8, 38*1.8)

# Dictionary of different source types 
SOURCE_TYPES = {0:"foreground", 1:"background"}

# Maximum number of each type of sources
NUM_FG_SOURCES = 1
NUM_BG_SOURCES = 3

# Init
pygame.init()

# Detine base font
font = pygame.font.SysFont('couriernew', 14) 

# Define default button style
BUTTON_STYLE = {"font" : font,
                "hover_color" : HOVER_DARK_GREY,
                "clicked_color" : BLUE,
                "clicked_font_color" : WHITE,
                "hover_font_color" : WHITE,
                "click_sound" : pygame.mixer.Sound("blipshort1.wav")}

# Define red button style
RED_BUTTON_STYLE = {"font" : font,
                "hover_color" : RED,
                "clicked_color" : ORANGE,
                "clicked_font_color" : WHITE,
                "hover_font_color" : WHITE,
                "click_sound" : pygame.mixer.Sound("blipshort1.wav")}

# Constructor for sources. Each source carries these attributes
class Source(pygame.sprite.Sprite):
    def __init__(self, xpos, ypos, idx, source_type):
        super(Source, self).__init__()
        # Possible image representations for sources
        self.images = [pygame.image.load("foreground.png").convert_alpha(),\
                        pygame.image.load("foreground_selected.png").convert_alpha(),\
                        pygame.image.load("background.png").convert_alpha(),\
                        pygame.image.load("background_selected.png").convert_alpha()]
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
        self.position = ((self.rect.center[0]- SOURCE_DISPLAY_WIDTH//2) / PIXELS_PER_METER, (DISPLAY_HEIGHT//2 - self.rect.center[1]) / PIXELS_PER_METER)
        # Angle in degrees
        self.angle = math.degrees(math.atan2(DISPLAY_HEIGHT//2 - self.rect.center[1], self.rect.center[0] - SOURCE_DISPLAY_WIDTH//2))      
        # Label associated with source instance
        self.text = SOURCE_TYPES[self.source_type] + " source: "+ ('%.4f' % self.position[0]) + ", " + ('%.4f' % self.position[1]) + \
                            "   Angle: " + ('%.4f' % self.angle)

# Intialize, render screen, start clock
screen = pygame.display.set_mode((SOURCE_DISPLAY_WIDTH + SPEC_DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("Speaker renderer")
clock = pygame.time.Clock()

# Exit flag
exit = False

# Initialize a group of sprites
source_list = []# pygame.sprite.Group()
clicked_on_nothing = False # If user clicks on nothing flag
pos_offset = (-1,-1) # Holds position of mouse pos to sprite pos offset for drag and drop
clicked_on_a_button = False # If user clicks on a button flag

# List of different potential status messages
statuses = ["bleb", \
        "Max number of background sources: " + str(NUM_BG_SOURCES), \
        "Foreground source cannot be deleted"]
# Current status to be displayed
current_status = statuses[0]


def change_color():
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

def change_source(source):
    print("hi")

def delete_source(source):
    global source_list
    # Remove source from source list as long as it is not a foreground source
    if not source.source_type == 0:
        source_list.remove(source)
    # Otherwise, let user know they cannot delete foreground source
    elif source.source_type == 0:
        global current_status
        current_status = statuses[2]

# Iterate through all selected sources and delete those that are selected
def delete_source_selected():
    global source_list
    for source in source_list:
        if source.selected:
            delete_source(source)
    
# Button definitions and positioning
new_source_button = Button((0,0,200,24), DARK_GREY, add_source, text="New background source", **BUTTON_STYLE)
new_source_button.rect.bottomleft = (screen.get_rect().bottomleft[0] + 5, screen.get_rect().bottomleft[1] - 5)

change_source_button = Button((0,0,130,24), DARK_GREY, change_color, text="Change source", **BUTTON_STYLE)
change_source_button.rect.bottomleft = (new_source_button.rect.bottomright[0] + 5, screen.get_rect().bottomleft[1] - 5)

delete_source_button = Button((0,0,120,24), DARK_RED, delete_source_selected, text="Delete source", **RED_BUTTON_STYLE)
delete_source_button.rect.bottomleft = (change_source_button.rect.bottomright[0] + 5, screen.get_rect().bottomleft[1] - 5)

button_list = [new_source_button, change_source_button, delete_source_button]

# Add one foreground source
add_source(x=randint(0.25*SOURCE_DISPLAY_WIDTH, 0.75*SOURCE_DISPLAY_WIDTH), y=randint(0.25*DISPLAY_HEIGHT, 0.75*DISPLAY_HEIGHT), source_type=0) 

# Main program loop
while not exit:
    # Get pygame  event
    for event in pygame.event.get():
        # Quit if pygame is quit
        if event.type == pygame.QUIT:
            exit = True

        # Button interaction logic
        new_source_button.check_event(event)
        change_source_button.check_event(event)
        delete_source_button.check_event(event)

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
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
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
                            #if not source.clicked:
                            #    source.selected = not source.selected
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
            if event.type == pygame.MOUSEBUTTONUP:
                for source in source_list:
                    source.clicked = False
                pos_offset = (-1,-1)

            # Update the clicked sprite's position for drag behavior
            for source in source_list:
                if source.clicked == True:
                    pos = pygame.mouse.get_pos()
                    source.rect.x = min(pos[0] - pos_offset[0], SOURCE_DISPLAY_WIDTH-(source.rect.width//2))
                    source.rect.y = pos[1] - pos_offset[1]
                    source.position = ((source.rect.center[0]- SOURCE_DISPLAY_WIDTH//2) / PIXELS_PER_METER, (DISPLAY_HEIGHT//2 - source.rect.center[1]) / PIXELS_PER_METER)
                    source.angle = math.degrees(math.atan2(DISPLAY_HEIGHT//2 - source.rect.center[1], source.rect.center[0] - SOURCE_DISPLAY_WIDTH//2))
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

        # Draw center indicator
        pygame.draw.circle(screen, DARK_GREY, (SOURCE_DISPLAY_WIDTH//2, DISPLAY_HEIGHT//2), 5)

        # Draw line separating source UI panel area from specgram panel
        pygame.draw.line(screen, DARK_GREY, (SOURCE_DISPLAY_WIDTH, 0), (SOURCE_DISPLAY_WIDTH, DISPLAY_HEIGHT), 5) 
      
        # Update button looks
        new_source_button.update(screen)
        change_source_button.update(screen)
        delete_source_button.update(screen)

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
        pygame.display.update()

        # TODO play sounds when clicked
        # for source in source_list:
        #     if source.selected:
        #         pygame.mixer.Sound(source)

        # 60 fps
        clock.tick(60)

# Quit cleanly if game is quit
pygame.quit()


                    