"""
Code to generate image using Genetic Algorithm
"""

from PIL import Image
import numpy as np
import math
from genetic_algorithm import GeneticAlgorithm as GA
import random
from joblib import Parallel, delayed

def fitness(X, y, weights):
	return np.array([np.sum(np.absolute(i-y)) for i in weights])

def mutation(chromosome , size):
	if random.randint(0, 1) == 1:
		for i in range(random.randint(0,size*10/100)):
			lower = 0
			upper = 255
			k = random.randint(0,255)
			v = random.randint(lower, upper)
			chromosome[k] = v
	return chromosome

def ga_parallel(row):
	ga = GA(initial_population=10, iterations=5000, fitness_func=fitness, max_population_size=200, \
		mutation_func=mutation, verbose=False, initialization_scheme="uniform", a=0, b=255, n_jobs=24, \
		maximize_cost=False)
	row_img = ga.fit(row, row)
	# row_img[row_img == 1] = 255
	return row_img

if __name__ == '__main__':
	img = np.array(Image.open("mona-200.bmp").convert("L"))

	final = Parallel(n_jobs=48, verbose=100, backend="multiprocessing")(delayed(ga_parallel)(row) for row in img)

	re_img = Image.fromarray(np.array(final).astype('uint8'))
	re_img.save("re_draw_grayscale_7.bmp")
