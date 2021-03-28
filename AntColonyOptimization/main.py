from random import random
from random import randint

def read_net(file_name):
    f = open(file_name, "r")
    net = {}
    n = int(f.readline())
    net["no_nodes"] = n
    graph = {}
    for i in range(n):
        line = f.readline()
        elems = line.split(",")
        graph[i] = {}
        for j in range(n):
            graph[i][j] = int(elems[j])
    net["graph"] = graph
    return net

class AntColonyOptimizer:
    def __init__(self, ants, evaporation_rate, intensification, choose_best, pheromone_left, alpha, beta):
        self.__ants = ants
        self.__evaporation_rate = evaporation_rate
        self.__intensification = intensification
        self.__choose_best = choose_best
        self.__pheromone_left = pheromone_left
        self.__alpha = alpha
        self.__beta = beta

        self.__no_nodes = None
        self.__distance_matrix = None
        self.__pheromone_matrix = None
        self.__probability_matrix = None
        self.__unvisited = None
        self.__best_path = None
        self.__best_score = None

    def initialize(self):
        self.__pheromone_matrix = {}
        self.__probability_matrix = {}

        for i in range(self.__no_nodes):
            self.__pheromone_matrix[i] = {}
            self.__probability_matrix[i] = {}
            for j in range(self.__no_nodes):
                if j != i:
                    self.__pheromone_matrix[i][j] = 1
                    self.__probability_matrix[i][j] = (1 / self.__distance_matrix[i][j])**self.__beta
                else:
                    self.__pheromone_matrix[i][j] = 0
                    self.__probability_matrix[i][j] = 0
        self.__unvisited = list(range(self.__no_nodes))

    def reset_unvisited(self):
        self.__unvisited = list(range(self.__no_nodes))

    def update_probabilities(self):
        for i in range(self.__no_nodes):
            for j in range(self.__no_nodes):
                if i != j:
                    self.__probability_matrix[i][j] = self.__pheromone_matrix[i][j]**self.__alpha * (1/self.__distance_matrix[i][j])**self.__beta

    def choose_next_node(self, current):
        probabilities = []

        total_probability = 0
        for i in self.__unvisited:
            total_probability += self.__probability_matrix[current][i]

        probability = 0
        for i in self.__unvisited:
            probabilities.append((i, probability + (self.__probability_matrix[current][i] / total_probability)))
            probability += self.__probability_matrix[current][i] / total_probability

        if random() < self.__choose_best:
            max_prob = 0
            node = 0
            for i in probabilities:
                if i[1] > max_prob:
                    max_prob = i[1]
                    node = i[0]
            return node
        else:
            chance = random()
            if probabilities[0][1] >= chance:
                return probabilities[0][0]

            for i in range(len(probabilities) - 1):
                if probabilities[i][1] < chance <= probabilities[i+1][1]:
                    return probabilities[i+1][0]

    def remove_node(self, node):
        for i in range(len(self.__unvisited)):
            if self.__unvisited[i] == node:
                self.__unvisited[i], self.__unvisited[-1] = self.__unvisited[-1], self.__unvisited[i]
                break
        self.__unvisited.pop()

    def evaluate(self, paths):
        best_score = 0
        best_path = None
        for path in paths:
            score = 0
            for i in range(len(path)-1):
                score += self.__distance_matrix[path[i]][path[i+1]]

            score = 1/score
            if score > best_score:
                best_path = path
                best_score = score
        return best_path, best_score

    def evaporation(self):
        for i in range(self.__no_nodes):
            for j in range(self.__no_nodes):
                self.__pheromone_matrix[i][j] = self.__pheromone_matrix[i][j] * (1-self.__evaporation_rate)


    def intensify(self, path):
        for i in range(len(path) - 1):
            self.__pheromone_matrix[path[i]][path[i+1]] += self.__intensification
        self.__pheromone_matrix[path[0]][path[-1]] += self.__intensification

    def fit(self, graph, no_nodes, iterations, early_stopping_count):
        self.__no_nodes = no_nodes
        self.__distance_matrix = graph
        num_equal = 0
        self.initialize()
        best_score_so_far = 0

        for i in range(iterations):
            paths = []
            path = []

            for ant in range(self.__ants):
                start_node = self.__unvisited[randint(0, len(self.__unvisited))-1]
                current_node = start_node
                while True:
                    path.append(current_node)
                    self.remove_node(current_node)
                    if len(self.__unvisited) != 0:
                        current_node_aux = self.choose_next_node(current_node)
                        self.__pheromone_matrix[current_node][current_node_aux] += self.__pheromone_left
                        current_node = current_node_aux
                    else:
                        break

                self.__pheromone_matrix[current_node][start_node] += self.__pheromone_left
                path.append(start_node)
                self.reset_unvisited()
                paths.append(path)
                path = []

            best_path, best_score = self.evaluate(paths)

            if best_score == best_score_so_far:
                num_equal += 1
            else:
                num_equal = 0

            if num_equal == early_stopping_count:
                print("Stopping early")
                break

            if i == 0:
                best_score_so_far = best_score
            else:
                if best_score > best_score_so_far:
                    best_score_so_far = best_score
                    self.__best_path = best_path
                    self.__best_score = best_score

            print(best_score_so_far)

            if num_equal == early_stopping_count:
                break

            self.evaporation()
            self.intensify(best_path)
            self.update_probabilities()

        return self.__best_path, self.__best_score



if __name__ == '__main__':
    file_name = "net.in"
    net = read_net(file_name)
    # ants, ev_rate, intensification, choose_best, pheromone, alpha, beta
    aco = AntColonyOptimizer(50, 0.5, 100, 0.2, 5, 4, 2)
    path, score = aco.fit(net['graph'], net['no_nodes'], 1000, 100)
    print("Best score: {} \n Best path: {}".format(score, path))
