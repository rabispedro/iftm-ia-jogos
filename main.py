import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()

font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
	RIGT = 1
	LEFT = 2
	UP = 3
	DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
	def __init__(self, w=640, h=480):
		self.w = w
		self.h = h

		self.display = pygame.display.set_mode((self.w, self.h))
		pygame.display.set_caption('IFTM | Snake IA')

	def play_step(self):
		for event in pygame.event.get():
			if(event.type == pygame.QUIT):
				pygame.quit()
				quit()


if __name__ == "__main__":
	game = SnakeGame()

	while True:
		game.play_step()
