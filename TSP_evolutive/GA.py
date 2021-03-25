from random import randint
from random import random
from Chromosome import Chromosome


class GA:
    def __init__(self, param=None, problParam=None):
        self.__param = param
        self.__problParam = problParam
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for _ in range(self.__param['popSize']):
            c = Chromosome(self.__problParam)
            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problParam['function'](c.repres, self.__problParam['graph'])

    def best_chromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if c.fitness < best.fitness:
                best = c
        return best

    def selection(self):
        total_fitness = 0
        for c in self.__population:
            total_fitness += 1/c.fitness

        probabilities = []
        probability = 0
        for c in self.__population:
            probabilities.append((c, probability + ((1/c.fitness) / total_fitness)))
            probability += (1/c.fitness) / total_fitness
        rand = random()

        if probabilities[0][1] >= rand:
            return probabilities[0][0]

        for i in range(len(probabilities)-1):
            if probabilities[i][1] < rand <= probabilities[i+1][1]:
                return probabilities[i+1][0]

    def one_generation(self):
        newPop = [self.best_chromosome()]
        for _ in range(self.__param['popSize'] - 1):
            p1 = self.selection()
            p2 = self.selection()
            off = p1.crossover(p2)
            rand = randint(1, 100)
            if rand <= self.__param['mut_rate']:
                off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()

    def medium_fitness(self):
        total_f = 0;
        for c in self.__population:
            total_f += c.fitness
        return total_f/self.__param['popSize']