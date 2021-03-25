from random import randint
from random import random


def read_net(file_name):
    f = open(file_name, "r")
    net = {}
    n = int(f.readline())
    net["no_nodes"] = n
    graph = {}
    for i in range(n):
        line = f.readline()
        elems = line.split(",")
        graph[i+1] = {}
        for j in range(n):
            graph[i+1][j+1] = int(elems[j])
    net["graph"] = graph
    return net


def get_random(v):
    n = len(v)
    index = randint(0, n-1)
    num = v[index]
    v[index], v[n-1] = v[n-1], v[index]
    v.pop()
    return num


def random_perm(n):
    v = [0] * n
    for i in range(n):
        v[i] = i + 1

    perm = []
    while len(v):
        perm.append(get_random(v))
    return perm


def fitness_funct(perm, graph):
    fitness = 0
    for i in range(len(perm)-1):
        fitness += graph[perm[i]][perm[i+1]]
    fitness += graph[perm[0]][perm[-1]]
    return fitness
