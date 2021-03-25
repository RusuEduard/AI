from utils import read_net, fitness_funct
from GA import GA


def main():
    file_name = "net.in"
    net = read_net(file_name)
    problParam = {'graph': net['graph'], 'function': fitness_funct, 'size': net['no_nodes']}
    param = {'popSize': 200, 'noGen': 5000, 'mut_rate': 50}
    ga = GA(param, problParam)
    ga.initialisation()
    ga.evaluation()
    g = 0
    graph = net['graph']
    while g < param['noGen']:
        g += 1
        ga.one_generation()
        chromo = ga.best_chromosome()
        print(chromo.fitness)
        #print(ga.medium_fitness())


main()
