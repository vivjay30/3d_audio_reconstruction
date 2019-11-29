""" 
Install dependencies: python3 -m pip install -U pygame --user

Small applet for bulding audio scenes

Be able to instantiate and place speakers for rendering SoundScenes(tm)

Current behavior:
Right click to add/delete sound source. Left click to select sound source and move it around.

"""
import sys
import pygame
import math

DISPLAY_HEIGHT = 600
DISPLAY_WIDTH = 600
BLACK = (0,0,0)
WHITE = (255,255,255)


class Item(pygame.sprite.Sprite):
    def __init__(self, xpos, ypos, idx):
        super(Item, self).__init__()
        self.image = pygame.image.load("speaker.png")
        self.clicked = False
        self.rect = self.image.get_rect()
        self.rect.x = xpos - self.rect.width//2
        self.rect.y = ypos - self.rect.height//2
        self.idx = idx
        self.text = "Source #" + str(self.idx) + ": " + \
                            str(self.rect.center[0]- DISPLAY_WIDTH//2) + ", " + str(DISPLAY_HEIGHT//2 - self.rect.center[1]) + \
                            "   Angle: " + str(math.degrees(math.atan2(DISPLAY_HEIGHT//2 - self.rect.center[1], self.rect.center[0] - DISPLAY_WIDTH//2)))



# Intialize, render screen, start clock
pygame.init()
screen = pygame.display.set_mode((DISPLAY_HEIGHT, DISPLAY_WIDTH))
pygame.display.set_caption("Speaker renderer")
clock = pygame.time.Clock()

# Exit flag
exit = False

# Initialize a group of sprites
item_list = []# pygame.sprite.Group()
text_list = []

font = pygame.font.SysFont('couriernew', 14) 


# Main program loop
while not exit:
    # Get pygame  event
    for event in pygame.event.get():
        # Quit if pygame is quit
        if event.type == pygame.QUIT:
            exit = True
        # If any mouse button is  pressed
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            x, y = pos[0], pos[1]

            # For right click
            if event.button == 3:
                something_there_already = False
                # For all sprites
                for item in item_list:
                    # If clicked, and something is there,  delete everything that's right-clicked
                    if item.rect.collidepoint(pos):
                        something_there_already = True
                        #text_list.remove(text)
                        item_list.remove(item)
                
                #  If there's nothing there, instantiate a new sprite
                if not something_there_already:
                    item_list.append(Item(x, y, len(item_list)+1)) # TODO: fix iteration scheme
                    # text = font.render("Mic #" + str(item_list[-1].idx) + ": " + \
                    #                     str(item_list[-1].rect.x) + ", " + str(item_list[-1].rect.y), \
                    #      True, BLACK, WHITE)
                    # text_list.append(text)
            
            # If left-clicked
            elif event.button == 1:
                # For the first item in the stack that's clicked, mark that it's clicked
                for item in item_list:
                    if item.rect.collidepoint(pos):
                        item.clicked = True
                        break

        # Mark all items as unclicked if no mouse button is pressed 
        if event.type == pygame.MOUSEBUTTONUP:
            for item in item_list:
                item.clicked = False

        ## Game logic
        # Update the clicked sprites position for drag behavior
        for item in item_list:
            if item.clicked == True:
                pos = pygame.mouse.get_pos()
                item.rect.x = pos[0] - (item.rect.width//2)
                item.rect.y = pos[1] - (item.rect.height//2)
                item.text = "Source #" + str(item.idx) + ": " + \
                            str(item.rect.center[0]- DISPLAY_WIDTH//2) + ", " + str(DISPLAY_HEIGHT//2 - item.rect.center[1]) + \
                            "   Angle: " + str(math.degrees(math.atan2(DISPLAY_HEIGHT//2 - item.rect.center[1], item.rect.center[0] - DISPLAY_WIDTH//2)))



        # Clear screen
        screen.fill(WHITE)

        i = 0

        pygame.draw.circle(screen, BLACK, (DISPLAY_WIDTH//2, DISPLAY_HEIGHT//2), 5) 

        for text in text_list:
            screen.blit(text, (3,3+i*14)) 
            i += 1

        # Draw items to the screen
        # item_list.draw(screen)
        i = 0
        for item in item_list:
            screen.blit(item.image, (item.rect.x,item.rect.y))
            text = font.render(item.text, True, BLACK, WHITE)
            screen.blit(text, (3,3+i*14))
            i += 1

        # Update the screen
        pygame.display.update()

        # 60 fps
        clock.tick(60)

pygame.quit()


                    