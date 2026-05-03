import neat
import csv
import os
import matplotlib.pyplot as plt
from main import eval_genomes as flappy_eval

# Lists to store plotting data
all_best_fitness = []
all_avg_fitness = []

class ExperimentReporter(neat.reporting.BaseReporter):
	def __init__(self, filename):
		self.filename = filename
		self.generation = 0
		os.makedirs(os.path.dirname(self.filename), exist_ok=True)
		with open(self.filename, mode='w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['generation', 'best_fitness', 'avg_fitness', 'species_count'])

	def end_generation(self, config, population, species_set):
		self.generation += 1
		all_fitness = [g.fitness for g in population.values() if g.fitness is not None]
		avg_fitness = sum(all_fitness) / len(all_fitness) if all_fitness else 0
		best_fitness = max(all_fitness) if all_fitness else 0
		
		# Store for plotting
		all_best_fitness.append(best_fitness)
		all_avg_fitness.append(avg_fitness)

		with open(self.filename, mode='a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([self.generation, best_fitness, avg_fitness, len(species_set.species)])
		
		print(f"Gen {self.generation} | Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f}")

def plot_fitness(game_name):
	plt.figure(figsize=(10, 6))
	gens = range(1, len(all_best_fitness) + 1)
	plt.plot(gens, all_best_fitness, 'r-', linewidth=2, label='Best Fitness')
	plt.plot(gens, all_avg_fitness, 'b-', linewidth=2, label='Average Fitness')
	plt.xlabel('Generation', fontsize=12)
	plt.ylabel('Fitness', fontsize=12)
	plt.title(f'NEAT {game_name} - Fitness Over Generations', fontsize=14)
	plt.legend(fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(f'{game_name.lower()}_fitness_plot.png', dpi=150)
	print(f"Saved: {game_name.lower()}_fitness_plot.png")
	plt.close()

def run_experiment(game_name, pop_size):
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward.txt')
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
	config.pop_size = pop_size

	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	p.add_reporter(ExperimentReporter(f"data/{game_name}_pop_{pop_size}.csv"))
	
	p.run(flappy_eval, 100) # Match Snake 100 generations
	plot_fitness(game_name)

if __name__ == "__main__":
	# Reset plotting lists
	all_best_fitness.clear()
	all_avg_fitness.clear()
	
	# Match Snake comparison trial
	run_experiment("FlappyBird", pop_size=150)