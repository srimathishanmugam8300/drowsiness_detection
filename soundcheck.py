import pygame
pygame.mixer.init()
pygame.mixer.music.load(r"D:\my_projects\drowsy_detection\src\two-tone-beeping-sound-422801.mp3")
pygame.mixer.music.play()
input("Press Enter to stop sound...")
pygame.mixer.music.stop()
