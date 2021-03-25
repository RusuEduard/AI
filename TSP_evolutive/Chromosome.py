from random import randint
from utils import random_perm


class Chromosome:
    def __init__(self, problParam=None):
        self.__problParam = problParam
        self.__repres = random_perm(problParam['size'])
        self.__fitness = 0.0

    @property
    def repres(self):
        return self.__repres

    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, rep=[]):
        self.__repres = rep

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def crossover(self, c):
        values = {}
        newrepres = []
        breakingPoint = randint(1, self.__problParam['size']-2)
        for i in range(breakingPoint):
            newrepres.append(self.__repres[i])
            values[self.__repres[i]] = 1
        for i in c.__repres:
            if i not in values:
                values[i] = 1
                newrepres.append(i)
        offspring = Chromosome(self.__problParam)
        offspring.repres = newrepres
        return offspring

    def mutation(self):
        pos1 = randint(0, len(self.__repres)-1)
        pos2 = randint(0, len(self.__repres)-1)
        self.__repres[pos1], self.__repres[pos2] = self.repres[pos2], self.repres[pos1]

    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__repres == other.__repres and self.__fitness == other.__fitness
