from sklearn.datasets import load_iris
import numpy as np
from numpy import random
from math import sqrt
import csv
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense



def loadIrisData():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featureNames = list(data['feature_names'])
    feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
    inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('petal length (cm)')]] for feat in inputs]
    return inputs, outputs, outputNames

def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def calculate_proximity(value1, value2):
    sum = 0
    for index in range(len(value1)):
        sum += (value2[index] - value1[index])**2
    return sqrt(sum)


def iris():
    inputs, outputs, outputNames = loadIrisData()
    index1 = random.randint(1, 50)
    index2 = random.randint(51, 100)
    index3 = random.randint(101, 150)

    index2_1 = random.randint(1, 150)
    index2_2 = random.randint(1, 150)
    index2_3 = random.randint(1, 150)
    u1 = inputs[index1]
    u2 = inputs[index2]
    u3 = inputs[index3]
    # u1 = inputs[index2_1]
    # u2 = inputs[index2_2]
    # u3 = inputs[index2_3]
    u = [u1, u2, u3]

    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    print(testOutputs)

    running = True
    iter = 0
    while running:
        clusters = [[], [], []]
        iter += 1
        # print("Iteratia: ", iter)
        correct = 0
        for index in range(len(trainInputs)):
            proxies = []
            for item in u:
                proxies.append(calculate_proximity(item, trainInputs[index]))
            for index2 in range(len(proxies)):
                if min(proxies) == proxies[index2]:
                    clusters[index2].append(trainInputs[index])
                    if trainOutputs[index] == index2:
                        correct += 1
        print("Accuracy: ", correct / len(trainInputs))

        cluster_outer_dist = []
        for index1 in range(len(u)):
            for index2 in range(index1 + 1, len(u)):
                cluster_outer_dist.append(calculate_proximity(u[index1], u[index2]))

        cluster_inner_dist = []
        for cluster in clusters:
            distances = []
            for index1 in range(len(cluster)):
                for index2 in range(index1 + 1, len(cluster)):
                    distances.append(calculate_proximity(cluster[index1], cluster[index2]))
            cluster_inner_dist.append(max(distances))

        print("Dunn index: ", min(cluster_outer_dist) / max(cluster_inner_dist))

        means = []
        for index in range(len(clusters)):
            cluster = clusters[index]
            mean = [0, 0]
            for el in cluster:
                mean[0] = mean[0] + el[0]
                mean[1] = mean[1] + el[1]
            mean[0] = mean[0] / len(cluster)
            mean[1] = mean[1] / len(cluster)
            means.append(mean)
        ok = 0
        for index in range(len(means)):
            if (means[index][1] - u[index][1]) == 0 and (means[index][0] - u[index][0]) == 0:
                ok = 1
            else:
                ok = 0
        if ok == 1:
            running = False
            break
        for index in range(len(means)):
            u[index] = means[index]

    test_clusters = [[], [], []]

    correct = 0
    for index in range(len(testInputs)):
        proxies = []
        for item in u:
            proxies.append(calculate_proximity(item, testInputs[index]))
        for index2 in range(len(proxies)):
            if min(proxies) == proxies[index2]:
                test_clusters[index2].append(testInputs[index])
                if testOutputs[index] == index2:
                    correct += 1

    print()
    print("Test Accuracy: ", correct / len(testOutputs))


def get_train_data(train_csv):
    with open(train_csv) as cvs_file:
        csv_reader = csv.reader(cvs_file, delimiter=',', quotechar='"')
        line_count = 0
        x_train = []
        y_train = []
        for row in csv_reader:

            if line_count == 0:
                line_count += 1
            else:
                x_train.append(row[1])
                if row[3] == 'negative':
                    y_train.append([1,0,0])
                elif row[3] == 'positive':
                    y_train.append([0,0,1])
                else:
                    y_train.append([0,1,0])
        return x_train, y_train


def get_test_data(test_csv):
    with open(test_csv) as cvs_file:
        csv_reader = csv.reader(cvs_file, delimiter=',', quotechar='"')
        line_count = 0
        x_test = []
        y_test = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                x_test.append(row[1])
                if row[2] == 'negative':
                    y_test.append([1,0,0])
                elif row[2] == 'positive':
                    y_test.append([0,0,1])
                else:
                    y_test.append([0,1,0])
        return x_test, y_test


def words_clustering():
    x_train, y_train = get_train_data('train.csv')
    x_test, y_test = get_test_data('test.csv')

    vectorizer = TfidfVectorizer(max_features=50)

    trainFeatures = vectorizer.fit_transform(x_train)
    testFeatures = vectorizer.transform(x_test)

    train = np.array(trainFeatures.toarray())
    test = np.array(testFeatures.toarray())

    # print(len(trainFeatures.toarray()))

    # tokenize = keras.preprocessing.text.Tokenizer(num_words=50)
    # tokenize.fit_on_texts(x_train)
    #
    # body_train = tokenize.texts_to_matrix(x_train)
    # body_test = tokenize.texts_to_matrix(x_test)

    model = Sequential()
    model.add(Dense(50, input_shape=(50,), activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train, np.array(y_train), epochs=100, validation_data=(test, np.array(y_test)), batch_size=50)

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    num_clusters = 3
    km = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        max_iter=300,
        random_state=0
    )

    km.fit(trainFeatures.toarray())

    print(km.inertia_)

    predictions = km.predict(testFeatures.toarray())
    correct = 0
    for index in range(len(predictions)):
        if y_test[index][predictions[index]] == 1:
            correct += 1
    print('Accuracy: ', correct/len(y_test))



if __name__ == '__main__':
    iris()
    words_clustering()
