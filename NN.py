import numpy as np
from prettytable import PrettyTable
from datetime import datetime


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def hardlim(x):
    return np.where(x >= 0.0, 1, -1)


class Perceptron:

    def __init__(self, random_state=1, shape=[None, 2]):

        self.random_state = random_state
        rgen = np.random.RandomState(self.random_state)

        self.weight = rgen.normal(loc=0.0, scale=0.01, size=shape[1])
        self.baias = rgen.normal(loc=0.0, scale=0.01, size=shape[1])[0]
        self.activation = sigmoid

    def updateWaights(self, delta_w):
        self.weight += delta_w * self.x

    def predict(self, x):
        self.x = x
        inp = np.dot(x, self.weight) + self.baias
        return self.activation(inp)



class NeuralNetworkLayer():

    def __init__(self, activation=hardlim, shape=[1, 2]):
        '''
        :param shape: [numberOfNodesInLayer, NumberOfInputsForEachNode]
        '''
        self.__nodes = []

        for _ in range(shape[0]):
            newP = Perceptron(shape=[None, shape[1]])
            newP.activation = activation
            self.__nodes.append(newP)

    def predict(self, x):
        '''
        this is list of neurons as layer with fully connected.

        :param x: a vector from previous layer as input.
        :return: a vector as output of all nodes.
        '''
        y = []
        for p in self.__nodes:
            y.append(p.predict(x))

        y = np.array(y)
        return y

    def updateWaights(self, delta_w):
        for i in range(self.__nodes.__len__()):
            self.__nodes[i].updateWaights(delta_w[i])


class NeuralNetwork():
    def __init__(self):
        self.epochs = 50
        self.learningRate = 0.1

        self.__startTime = None
        self.time = None
        self.layers = []
        self.errors = []

    def addLayer(self, activation=hardlim, shape=[1, 2]):
        '''
        adding new layer to network.

        :param activation: the activation function for all nodes in this layer.
        :param shape: [numberOfNodesInLayer, NumberOfInputsForEachNode]
        :return:
        '''

        newLayer = NeuralNetworkLayer(activation=activation, shape=shape)
        self.layers.append(newLayer)

    def predict(self, x):
        y = x
        for layer in self.layers:
            y = layer.predict(y)

        return y

    def fit(self, x, y):
        self.__startTime = datetime.now()
        for epoch in range(self.epochs):
            error = np.zeros(y.shape[1])
            for xi, t in zip(x, y):
                yi = self.predict(xi)

                # last layer error
                delta = yi * (1 - yi) * (t - yi)
                delta_w = self.learningRate * delta * yi
                self.layers[self.layers.__len__() - 1].updateWaights(delta_w)

                # other layers error

                error += abs(t - yi)

            e = 0
            for i in error:
                e += i
            e = e/error.__len__()
            e = e / y.__len__()

            if epoch % 10 == 0:
                print('epoch:{} | error:{}'.format(epoch, e))

            # print(e)
            self.errors.append(e)
        self.time = datetime.now() - self.__startTime

    def test(self, x, t):
        num_error = 0
        e = 0
        for i in range(x.__len__()):
            y = self.predict(x[i])
            # if y != t[i]:
            #     num_error += 1
            e += abs(t[i] - y)
            sum_of_e = 0
            for i in e:
                sum_of_e += i

            e = sum_of_e/e.__len__()

        self.error = e / x.__len__()

    def summry(self):
        table = PrettyTable(['shape', 'running time', 'accuracy', 'epochs', 'learning rate'])
        accuracy = int(100 * (1 - self.error))
        table.add_row(['-', '{}'.format(self.time), '{}%'.format(accuracy), self.epochs, self.learningRate])
        print(table)

    def drawOut(self):
        pass