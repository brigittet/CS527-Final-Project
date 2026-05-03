import pygame
import random
import time
from pipe import Pipe
from bird import Bird
from settings import WIDTH, HEIGHT, pipe_size, pipe_gap, pipe_pair_sizes

class World:
	def __init__(self, screen):
		self.screen = screen
		# Pure randomness: seeded by current time
		random.seed(time.time())
		
		self.world_shift = -6 
		self.gravity = 0.5
		self.current_pipe = None
		self.pipes = pygame.sprite.Group() 
		self.player = pygame.sprite.Group() 
		self._generate_world()

	def _generate_world(self):
		self._add_pipe()

	def _add_pipe(self):
		pipe_pair_size = random.choice(pipe_pair_sizes)
		top_h, bot_h = pipe_pair_size[0] * pipe_size, pipe_pair_size[1] * pipe_size

		pipe_top = Pipe((WIDTH, 0 - (bot_h + pipe_gap)), pipe_size, HEIGHT, True)
		pipe_bottom = Pipe((WIDTH, top_h + pipe_gap), pipe_size, HEIGHT, False)
		self.pipes.add(pipe_top)
		self.pipes.add(pipe_bottom)
		
		if self.current_pipe is None:
			self.current_pipe = pipe_top

	def _handle_collisions(self):
		for bird in self.player:
			if pygame.sprite.spritecollide(bird, self.pipes, False) or \
			   bird.rect.bottom >= HEIGHT or bird.rect.top <= 0:
				bird.kill()

	def update(self, jump_list):
		self.pipes.update(self.world_shift)

		rightmost_pipe_x = 0
		for pipe in self.pipes:
			if pipe.rect.right > rightmost_pipe_x:
				rightmost_pipe_x = pipe.rect.right
		
		if rightmost_pipe_x < WIDTH - 250: 
			self._add_pipe()

		for pipe in self.pipes:
			if pipe.rect.right > (WIDTH // 2 - pipe_size):
				self.current_pipe = pipe
				break

		for i, bird in enumerate(self.player.sprites()):
			bird.direction.y += self.gravity
			bird.rect.y += bird.direction.y
			bird.update(jump_list[i])

		self._handle_collisions()
		
		self.pipes.draw(self.screen)
		self.player.draw(self.screen)