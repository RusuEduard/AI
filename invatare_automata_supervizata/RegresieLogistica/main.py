import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from math import log2


def load_data(file_name):
    inputs = []
    outputs = []
    f = open(file_name, 'r')
    for i in range(150):
        line = f.readline()
        line = line[:-1]
        line = line.split(',')
        inputs.append([float(nr) for nr in line[:-1]])
        outputs.append(line[-1])
    return inputs, outputs


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BinaryClassifier:
    def __init__(self, label1, label2, train_data, valid_data):
        self.__weights = [0] * (len(train_data[0]) - 1)
        self.__labels = {1: label1, 2: label2}
        self.__train_data = None
        self.__valid_data = None
        self.init_data(train_data, valid_data)

    def init_data(self, input, output):
        count = 0
        self.__train_data = []
        for list in input:
            self.__train_data.append([])
            for el in list[:-1]:
                self.__train_data[-1].append(el)
            if list[-1] == self.__labels[1]:
                self.__train_data[-1].append(1)
            else:
                self.__train_data[-1].append(0)

        self.__valid_data = []
        for list in output:
            self.__valid_data.append([])
            for el in list[:-1]:
                self.__valid_data[-1].append(el)
            if list[-1] == self.__labels[1]:
                self.__valid_data[-1].append(1)
            else:
                self.__valid_data[-1].append(0)

        # print(count)
        # for el in self.__valid_data:
        #     print(el)
        # print(" ")
        # for el in self.__train_data:
        #     print(el)
        # # [[x1], [x2], [x3], [y]]
        # data = []
        # for _ in range(len(input[0]) + 1):
        #     data.append([])
        # for index in range(len(input)):
        #     for j in range(len(data) - 1):
        #         data[j].append(input[index][j])
        #     if output[index] == self.__labels[1]:
        #         data[-1].append(1)
        #     else:
        #         data[-1].append(0)
        #
        # # print(data[-1])
        #
        # for index in range(len(data) - 1):
        #     # self.plotDataHistogram(data[index], str(index) + " before")
        #     data[index] = self.z_standardisation(data[index])
        #     # self.plotDataHistogram(data[index], str(index) + " after")
        #
        # indexes = [i for i in range(len(input))]
        # train_sample = np.random.choice(indexes, int(0.8 * len(input)), replace=False)
        # validation_sample = [i for i in indexes if not i in train_sample]
        #
        # count = 0
        # self.__train_data = []
        # for i in train_sample:
        #     self.__train_data.append([1])
        #     for elem in data:
        #         self.__train_data[-1].append(elem[i])
        #     if data[-1][i] == 1:
        #         count += 1
        #
        # self.__valid_data = []
        # for i in validation_sample:
        #     self.__valid_data.append([1])
        #     for elem in data:
        #         self.__valid_data[-1].append(elem[i])

    def z_standardisation(self, data):
        sum = 0
        for el in data:
            sum += el
        mean = sum / len(data)
        sum = 0
        for el in data:
            sum += (mean - el) ** 2
        std_dev = sqrt(sum / len(data))

        print("mean", mean)
        print("std_dev", std_dev)

        new_data = []
        for i in data:
            new_data.append((i - mean) / std_dev)
        return new_data

    def normalize_data(self, data):
        new_data = []
        for i in data:
            new_data.append((i - min(data)) / (max(data) - min(data)))
        return new_data

    def plotDataHistogram(self, x, variableName):
        n, bins, patches = plt.hist(x, 10)
        plt.title('Histogram of ' + variableName)
        plt.show()

    def get_train(self):
        return self.__train_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def calc_error(self, data):
        computed = 0
        for i in range(len(self.__weights)):
            computed += data[i] * self.__weights[i]
        error = (data[-1] - sigmoid(computed))*(sigmoid_derivative(sigmoid(computed)))
        return error

    def gradient_descent(self, learning_rate, iterations):
        for iteration in range(iterations):
            step = [0] * (len(self.__weights))
            for i in range(len(self.__train_data)):
                error = self.calc_error(self.__train_data[i])
                for j in range(len(step)):
                    step[j] += self.__train_data[i][j] * error
            for i in range(len(self.__weights)):
                self.__weights[i] = self.__weights[i] - learning_rate * step[i] / len(self.__train_data)

            # loss = 0
            # for i in range(len(self.__train_data)):
            #     computed = 0
            #     for j in range(len(self.__weights)):
            #         computed += self.__weights[j]*self.__train_data[i][j]
            #     if self.__train_data[i][-1] == 1:
            #         loss += -log2(1 - sigmoid(computed))
            # print(loss)

    def validate(self):
        result = []
        for iteration in range(len(self.__valid_data)):
            computed = 0
            for i in range(len(self.__weights)):
                computed += self.__weights[i] * self.__valid_data[iteration][i]
            computed = sigmoid(computed)
            result.append(computed)

        return result


class MultiClass:
    def __init__(self, label1, label2, label3, input_data, output_data):
        self.__train_data = None
        self.__valid_data = None
        self.__label1 = label1
        self.__label2 = label2
        self.__label3 = label3
        self.init_data(input_data, output_data)

    def init_data(self, input_data, output_data):
        # [[x1], [x2], [x3], [y]]
        data = []
        for _ in range(len(input_data[0]) + 1):
            data.append([])
        for index in range(len(input_data)):
            for j in range(len(data) - 1):
                data[j].append(input_data[index][j])
            data[-1].append(output_data[index])
        # print(data[-1])

        for index in range(len(data) - 1):
            # self.plotDataHistogram(data[index], str(index) + " before")
            data[index] = self.z_standardisation(data[index])
            # self.plotDataHistogram(data[index], str(index) + " after")

        indexes = [i for i in range(len(input_data))]
        train_sample = np.random.choice(indexes, int(0.8 * len(input_data)), replace=False)
        validation_sample = [i for i in indexes if not i in train_sample]

        count = 0
        self.__train_data = []
        for i in train_sample:
            self.__train_data.append([1])
            for elem in data:
                self.__train_data[-1].append(elem[i])
            if data[-1][i] == 1:
                count += 1

        self.__valid_data = []
        for i in validation_sample:
            self.__valid_data.append([1])
            for elem in data:
                self.__valid_data[-1].append(elem[i])

    def z_standardisation(self, data):
        sum = 0
        for el in data:
            sum += el
        mean = sum / len(data)
        sum = 0
        for el in data:
            sum += (mean - el) ** 2
        std_dev = sqrt(sum / len(data))

        print("mean", mean)
        print("std_dev", std_dev)

        new_data = []
        for i in data:
            new_data.append((i - mean) / std_dev)
        return new_data

    def train(self, lerning_rate, iterations):
        train = list(self.__train_data)
        valid = list(self.__valid_data)
        print("loss 1")
        class1 = BinaryClassifier(self.__label1, "other", train, valid)
        class1.gradient_descent(lerning_rate, iterations)
        # print(self.__label1)
        result1 = class1.validate()
        print(" ")

        print("loss 2")
        class2 = BinaryClassifier(self.__label2, "other", train, valid)
        class2.gradient_descent(lerning_rate+0.1, iterations)
        # print(self.__label2)
        result2 = class2.validate()
        print(" ")
        #
        print("loss 3")
        class3 = BinaryClassifier(self.__label3, "other", train, valid)
        class3.gradient_descent(lerning_rate, iterations)
        print(self.__label3)
        result3 = class3.validate()
        print(" ")
        #
        results = []
        for index in range(len(result1)):
            results.append([result1[index], result2[index], result3[index]])

        # for res in results:
        #     print(res)

        corecte = 0
        for index in range(len(results)):
            answ = min(results[index])
            if answ == results[index][0] and self.__valid_data[index][-1] == self.__label1:
                corecte += 1
            if answ == results[index][1] and self.__valid_data[index][-1] == self.__label2:
                corecte += 1
            if answ == results[index][2] and self.__valid_data[index][-1] == self.__label3:
                corecte += 1
        print("Accuracy: ", corecte/len(results))

def print_hi():
    x = "1,2,3,ana"
    y = [int(nr) for nr in x.split(',')[:-1]]
    print(y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inputs, outputs = load_data("iris.data")
    classifier = MultiClass("Iris-setosa", "Iris-versicolor", "Iris-virginica", inputs, outputs)
    classifier.train(0.01, 10000)

