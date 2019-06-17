"""
Genetic Algorithm
"""
import random
import numpy as np
import heapq
from itertools import product

class GeneticAlgorithm:
	def __init__(self, initial_population=6, min_population_size=2, max_population_size=6, \
		iterations=10, fitness_func=None, mutation_func=None, crossover_func=None, \
		random_seed=None, maximize_cost=True, n_jobs=-1, verbose=True, initialization_scheme="normal", \
		a=0, b=10):

		self.__INITIAL_POPULATION = initial_population
		self.__MIN_POPULATION_SIZE = min_population_size # TODO
		self.__MAX_POPULATION_SIZE = max_population_size
		self.__ITERATIONS = iterations
		self.__FITNESS_FUNC = fitness_func if fitness_func else self.__getDefaultFitnessFunc
		self.__MUATION_FUNC = mutation_func if mutation_func else self.__getDefaultMutationFunc
		self.__CROSSOVER_FUNC = crossover_func if crossover_func else self.__getDefaultCrossoverFunc
		self.__RANDOM_SEED = random_seed
		self.__MAXIMIZE_COST = maximize_cost
		self.__N_JOBS = n_jobs
		self.__VERBOSE = verbose
		self.__INIT_SCHEME = initialization_scheme
		self.__A = a
		self.__B = b
		self.select_n_fittest = self.__INITIAL_POPULATION//2
		self.weights = None
		self.best_fit = None

	def __population(self, size):
		if self.__RANDOM_SEED:
			random.seed(self.__RANDOM_SEED)
		if self.__INIT_SCHEME == "normal":
			self.weights = np.random.normal(self.__A, self.__B, (self.__INITIAL_POPULATION, size))
		elif self.__INIT_SCHEME == "uniform":
			self.weights = np.random.uniform(self.__A, self.__B, (self.__INITIAL_POPULATION, size))

	def __selection(self, X, y, size):
		fitness_score = self.__FITNESS_FUNC(X, y, self.weights)
		best_fit_pool = []
		for idx, score in enumerate(fitness_score):
			if self.__MAXIMIZE_COST:
				heapq.heappush(best_fit_pool, (-1*score, idx))
			else:
				heapq.heappush(best_fit_pool, (score, idx))

		self.weights = [self.weights[heapq.heappop(best_fit_pool)[1]] for _ in range(self.select_n_fittest)]
		self.best_fit = self.weights[0]

		population_size = len(self.weights)
		new_weights = []

		cnt = 0
		for i, j in product(range(population_size), range(population_size)):
			if i == j:
				continue
			if cnt > self.__MAX_POPULATION_SIZE/2:
				break
			cnt += 1
			new_weights.append(self.__CROSSOVER_FUNC(self.weights[i], self.weights[j], size))

		self.weights = np.array(new_weights+self.weights)
		self.select_n_fittest = self.weights.shape[0]//2

	def __getDefaultFitnessFunc(self, *args):
		def fitness_error(X, y, weights):
			return 1 // (1 + abs(np.dot(weights, X)-y))
		return fitness_error(args[0], args[1], args[2])

	def __getDefaultCrossoverFunc(self, *args):
		def crossover(chromosome1, chromosome2, size):
			k = size//2
			return self.__MUATION_FUNC(np.concatenate((chromosome1[:k],chromosome2[k:]), axis=None), size)
		return crossover(args[0], args[1], args[2])

	def __getDefaultMutationFunc(self, *args):
		def mutation(chromosome, size):
			if self.__RANDOM_SEED:
				random.seed(self.__RANDOM_SEED)
			k = random.randint(0, size-1)
			chromosome[k] = 0.1 + chromosome[k]//2
			return chromosome
		return mutation(args[0], args[1])

	def fit(self, X, y):
		size = X.shape[0]
		self.__population(size)
		for i in range(0, self.__ITERATIONS):
			if self.__VERBOSE:
				pop_size = self.weights.shape[0]
				print("Iteration : {}, population size : {}, ".format(i, pop_size)),
		
			self.__selection(X, y, size)
		
			if self.__VERBOSE:
				best_val = np.dot(X, self.best_fit)
				best_score = best_val-y
				print("best score so far : {}, closest val : {}".format(best_score, -1*best_val))
		return self.best_fit

if __name__ == '__main__':
	ga = GeneticAlgorithm(initial_population=10, iterations=5000, verbose=True, max_population_size=10000)
	print(ga.fit(np.array([1, 2, 3, 4, 5, 6, 11, 23, 45, 1, 23, 20]), 51))
