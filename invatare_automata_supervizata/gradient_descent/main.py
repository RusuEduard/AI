import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


# loads data from csv file
def load_data(file_name, input_variable1, input_variable2, input_variable3, output_variable):
    data = []
    data_names = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                data_names = row
            else:
                data.append(row)
            line_count += 1

    selected_variable1 = data_names.index(input_variable1)
    inputs1 = [float(data[i][selected_variable1]) for i in range(len(data))]

    selected_variable2 = data_names.index(input_variable2)
    inputs2 = [float(data[i][selected_variable2]) for i in range(len(data))]

    selected_variable3 = data_names.index(input_variable3)
    inputs3 = [float(data[i][selected_variable3]) for i in range(len(data))]

    selected_output = data_names.index(output_variable)
    outputs = [float(data[i][selected_output]) for i in range(len(data))]

    return inputs1, inputs2, inputs3, outputs


def calculate_error(input_data, w):
    error = 0
    for i in range(len(w)):
        error += input_data[i] * w[i]
    error = error - input_data[-1]
    return error


def stochastic_gradient_descent(input_data, learning_rate, valid_data):
    w = [0] * (len(input_data[0]) - 1)
    for i in range(len(input_data)):
        error = calculate_error(input_data[i], w)
        for j in range(len(w)):
            w[j] = w[j] - learning_rate * error * input_data[i][j]
        # print(mean_square_error(input_data, w))
    return w


def plotModel(trainInputs1, trainInputs2, trainOutputs, w, title):
    xref = np.linspace(min(trainInputs1),max(trainInputs1),1000)
    yref = np.linspace(min(trainInputs2),max(trainInputs2),1000)
    x_surf,y_surf = np.meshgrid(xref,yref)
    zref=[]
    for el2 in range(len(yref)):
        for el in range(len(xref)):
            zref.append([w[0] + w[1] * xref[el] + w[2] * yref[el2]])
    z_vals = np.array(zref)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(trainInputs1,trainInputs2,trainOutputs, label = 'train data')
    ax.plot_surface(x_surf,y_surf,z_vals.reshape(x_surf.shape),color = 'None', alpha=0.3)
    plt.legend()
    plt.xlabel('gdp capita')
    plt.ylabel('freedom')
    plt.title(title)
    plt.show()


def batch_gradient_descent(input_data, learning_rate, iterations, valid_data):
    w = [0] * (len(input_data[0]) - 1)
    for iteration in range(iterations):
        step = [0] * (len(input_data[0]) - 1)
        for i in range(len(input_data)):
            error = calculate_error(input_data[i], w)
            for j in range(len(step)):
                step[j] += input_data[i][j] * error
        for i in range(len(w)):
            w[i] = w[i] - learning_rate * step[i] / len(input_data)
        print(mean_square_error(valid_data, w))
    return w


def mean_square_error(input_data, w):
    error = 0
    for i in input_data:
        error += (calculate_error(i, w)) ** 2
    return sqrt(error / len(input_data))


# min-max scaling
def normalize_data(data):
    new_data = []
    for i in data:
        new_data.append((i - min(data)) / (max(data) - min(data)))
    return new_data


def z_standardisation(data):
    sum = 0
    for el in data:
        sum += el
    mean = sum/len(data)
    sum = 0
    for el in data:
        sum += (mean - el)**2
    std_dev = sqrt(sum/len(data))

    new_data = []
    for i in data:
        new_data.append((i - mean)/std_dev)
    return new_data



def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input1, input2, input3, output = load_data("2017.csv", 'Economy..GDP.per.Capita.', "Freedom", "Health..Life.Expectancy.", 'Happiness.Score')

    np.random.seed(5)
    indexes = [i for i in range(len(input1))]
    train_sample = np.random.choice(indexes, int(0.8 * len(input1)), replace=False)
    validation_sample = [i for i in indexes if not i in train_sample]

    trainX1 = [input1[i] for i in train_sample]
    trainX2 = [input2[i] for i in train_sample]
    trainX3 = [input3[i] for i in train_sample]
    trainY = [output[i] for i in train_sample]

    validX1 = [input1[i] for i in validation_sample]
    validX2 = [input2[i] for i in validation_sample]
    validX3 = [input3[i] for i in validation_sample]
    validY = [output[i] for i in validation_sample]

    trainX1_norm = normalize_data(trainX1)
    trainX2_norm = normalize_data(trainX2)
    trainX3_norm = normalize_data(trainX3)
    trainY_norm = normalize_data(trainY)

    validX1_norm = normalize_data(validX1)
    validX2_norm = normalize_data(validX2)
    validX3_norm = normalize_data(validX3)
    validY_norm = normalize_data(validY)

    trainX1_z = z_standardisation(trainX1)
    trainX2_z = z_standardisation(trainX2)
    trainY_z = z_standardisation(trainY)

    validX1_z = z_standardisation(validX1)
    validX2_z = z_standardisation(validX2)
    validY_z = z_standardisation(validY)

    input_data = []
    for i in range(len(trainX1)):
        input_data.append([1, trainX1[i], trainX2[i], trainX3[i], trainY[i]])

    valid_data = []
    for i in range(len(validX1)):
        valid_data.append([1, validX1[i], validX2[i], validX3[i], validY[i]])

    input_data_norm = []
    for i in range(len(trainX1_norm)):
        input_data_norm.append([1, trainX1_norm[i], trainX2_norm[i], trainX3_norm[i], trainY_norm[i]])

    input_data_z = []
    for i in range(len(trainX1_z)):
        input_data_z.append([1, trainX1_z[i], trainX2_z[i], trainY_z[i]])

    valid_data_z = []
    for i in range(len(validX1_z)):
        valid_data_z.append([1, validX1_z[i], validX2_z[i], validY_z[i]])

    valid_data_norm = []
    for i in range(len(validX1_norm)):
        valid_data_norm.append([1, validX1_norm[i], validX2_norm[i], validX3_norm[i], validY_norm[i]])

    # w = stochastic_gradient_descent(input_data, 0.01, valid_data)
    # # print("Stochastic error: ", mean_square_error(valid_data, w))
    # plotModel(trainX1, trainX2, trainY, w, 'Stochastic on original data')
    # w = stochastic_gradient_descent(input_data_norm, 0.01, valid_data_norm)
    # # print("Stochastic err on min-max norm: ", mean_square_error(valid_data_norm, w))
    # plotModel(trainX1_norm, trainX2_norm, trainY_norm, w, 'Stochastic on min-max')
    # w = stochastic_gradient_descent(input_data_z, 0.01, valid_data_z)
    # # print("Stochastic err on z_norm: ", mean_square_error(valid_data_z, w))
    # plotModel(trainX1_z, trainX2_z, trainY_z, w, 'Stochastic on z-norm')

    print(" ")
    print(" ")
  #  w = batch_gradient_descent(input_data, 0.01, 1000, valid_data)
    # print("batch error: ", mean_square_error(valid_data, w))
    # plotModel(trainX1, trainX2, trainY, w, 'batch on original')

    w = batch_gradient_descent(input_data_norm, 0.001, 3000, valid_data_norm)
    print(w)
    # plotModel(trainX1_norm, trainX2_norm, trainY_norm, w, 'batch on min-max')
#
  #  w = batch_gradient_descent(input_data_z, 0.01, 1000, valid_data_z)
    # print("batch error on z_norm: ", mean_square_error(valid_data_z, w))
    # plotModel(trainX1_z, trainX2_z, trainY_z, w, 'batch on z-norm')
