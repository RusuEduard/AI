from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import neural_network
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization

def loadIrisData():
    from sklearn.datasets import load_iris

    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featureNames = list(data['feature_names'])
    feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
    inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('petal length (cm)')]] for feat in inputs]
    return inputs, outputs, outputNames


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


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

def training(classifier, trainInputs, trainOutputs):
    # step4: training the classifier
    # identify (by training) the classification model
    classifier.fit(trainInputs, trainOutputs)
    loss_values = classifier.loss_curve_
    plt.plot(loss_values)
    plt.show()


def classification(classifier, testInputs):
    # step5: testing (predict the labels for new inputs)
    # makes predictions for test data
    computedTestOutputs = classifier.predict(testInputs)

    return computedTestOutputs


def evalMultiClass(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix

    confMatrix = confusion_matrix(realLabels, computedLabels)
    acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
        recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
    return acc, precision, recall, confMatrix


def data2FeaturesMoreClasses(inputs, outputs, outputNames):
    labels = set(outputs)
    noData = len(inputs)
    for crtLabel in labels:
        x = [inputs[i][0] for i in range(noData) if outputs[i] == crtLabel ]
        y = [inputs[i][1] for i in range(noData) if outputs[i] == crtLabel ]
        plt.scatter(x, y, label=outputNames[crtLabel])
    plt.xlabel('feat1')
    plt.ylabel('feat2')
    plt.legend()
    plt.show()

def iris():
    inputs, outputs, outputNames = loadIrisData()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

    # plot the training data distribution on classes
    plt.hist(trainOutputs, 3, rwidth=0.8)
    plt.xticks(np.arange(len(outputNames)), outputNames)
    plt.show()

    # plot the data in order to observe the shape of the classifier required in this problem
    data2FeaturesMoreClasses(trainInputs, trainOutputs, outputNames)

    # normalise the data
    trainInputs, testInputs = normalisation(trainInputs, testInputs)

    # liniar classifier and one-vs-all approach for multi-class
    # classifier = linear_model.LogisticRegression()

    # non-liniar classifier and softmax approach for multi-class
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100, solver='sgd',
                                              verbose=10, random_state=1, learning_rate_init=.1)

    training(classifier, trainInputs, trainOutputs)
    predictedLabels = classification(classifier, testInputs)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)


def load_data_img():
    x_data = []
    y_data = []
    IMG_SIZE = 200
    path_jpgs = r".\Images"
    path_sepia = r".\sepia"
    index = 0
    for img in os.listdir(path_jpgs):
        try:
            index += 1
            img_arr = image.load_img(os.path.join(path_jpgs, img), target_size=(IMG_SIZE, IMG_SIZE, 3))
            img_arr = image.img_to_array(img_arr)
            img_arr = img_arr / 255.0
            x_data.append(img_arr)
            y_data.append([0, 1])
            if index == 1000:
                break
        except Exception as ex:
            pass
    print(len(x_data))
    index = 0
    for img in os.listdir(path_sepia):
        try:
            index += 1
            img_arr = image.load_img(os.path.join(path_sepia, img), target_size=(IMG_SIZE, IMG_SIZE, 3))
            img_arr = image.img_to_array(img_arr)
            img_arr = img_arr / 255.0
            x_data.append(img_arr)
            y_data.append([1, 0])
            if index == 1000:
                break
        except Exception as ex:
            pass
    print(len(x_data))
    return np.array(x_data), np.array(y_data)


def sepia(source_name, result_name):
    path_jpgs = r".\Images"
    file_path = os.path.join(path_jpgs, result_name)
    source = Image.open(source_name)
    result = Image.new('RGB', source.size)
    for x in range(source.size[0]):
        for y in range(source.size[1]):
            r, g, b = source.getpixel((x, y))
            red = int(r * 0.393 + g * 0.769 + b * 0.189)
            green = int(r * 0.349 + g * 0.686 + b * 0.168)
            blue = int(r * 0.272 + g * 0.534 + b * 0.131)
            result.putpixel((x, y), (red, green, blue))
    result.save(result_name, "JPEG")


def sepia_model():
    IMG_SIZE = 200
    x_data, y_data = load_data_img()

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=20, test_size=0.3)
    sepia_count = 0
    norm_count = 0
    for i in y_train:
        if i[0] == 1:
            sepia_count += 1
        else:
            norm_count += 1

    print(sepia_count, norm_count)

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    #
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test), batch_size=75)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    img1 = image.load_img("poza1.jpg", target_size=(IMG_SIZE, IMG_SIZE, 3))
    img1 = image.img_to_array(img1)
    img1 = img1/255.
    plt.imshow(img1)
    img1 = np.expand_dims(img1, axis=0)

    img2 = image.load_img("sepia1641.jpg", target_size=(IMG_SIZE, IMG_SIZE, 3))
    img2 = image.img_to_array(img2)
    img2 = img2 / 255.
    plt.imshow(img2)
    img2 = np.expand_dims(img2, axis=0)

    prob1 = model.predict(img1)
    prob2 = model.predict(img2)
    print(prob1)
    print(prob2)



if __name__ == '__main__':
    iris()

