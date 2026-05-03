import pygame, os, neat, sys
from settings import WIDTH, HEIGHT, ground_space, pipe_size
from world import World
from bird import Bird

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + ground_space))

def eval_genomes(genomes, config):
	num_games = 3
	genome_list = [g for _, g in genomes]
	total_fitness = [0.0] * len(genome_list)

	for game_idx in range(num_games):
		world = World(screen)
		nets = []
		ge = []
		birds = []
		penalized = [False] * len(genome_list) # Track who we already penalized
		
		for genome in genome_list:
			net = neat.nn.FeedForwardNetwork.create(genome, config)
			nets.append(net)
			new_bird = Bird((WIDTH//2 - pipe_size, HEIGHT//2 - pipe_size), 30)
			world.player.add(new_bird)
			birds.append(new_bird)
			ge.append(genome)

		clock = pygame.time.Clock()

		while True:
			birds_alive = world.player.sprites()
			if len(birds_alive) == 0:
				break 

			passed_pipe = False
			if 0 <= (world.current_pipe.rect.centerx - WIDTH // 2) < 6:
				passed_pipe = True

			jump_list = []
			for i, bird in enumerate(birds_alive):
				orig_idx = birds.index(bird)
				
				if passed_pipe:
					total_fitness[orig_idx] += 100 
				
				# Inputs scaled for stability with ReLU
				output = nets[orig_idx].activate((
					bird.rect.y / HEIGHT, 
					abs(bird.rect.x - world.current_pipe.rect.x) / WIDTH, 
					world.current_pipe.rect.bottom / HEIGHT, 
					(world.current_pipe.rect.bottom + 160) / HEIGHT
				))
				jump_list.append(output[0] > 0.5)

			world.update(jump_list)

			# Correct Death Penalty Logic
			for i, bird in enumerate(birds):
				if not bird.alive() and not penalized[i]:
					total_fitness[i] -= 1.2
					penalized[i] = True # Only penalize once per round

		pygame.display.update()

	# Set the final average fitness for NEAT
	for i, genome in enumerate(genome_list):
		genome.fitness = max(0, total_fitness[i] / num_games)

def run(config_path):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	p.run(eval_genomes, 100)

if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward.txt')
	run(config_path)