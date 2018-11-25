from NN import *
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

def main():

    # reading dataset
    print('Reading dataset ...', end='')
    mndata = MNIST('./train')
    train_images, train_labels = mndata.load_training()

    mndata = MNIST('./test')
    test_images, test_labels = mndata.load_testing()
    print('      |DONE|')


    # pre processing dataset
    print('Pre processing ...', end='')
    X_train = np.array(train_images)
    X_train = normalize(X_train)
    y_train = np.array(train_labels)

    X_test = np.array(test_images)
    X_test = normalize(X_test)
    y_test = np.array(test_labels)

    y_train = encode(y_train)
    y_test = encode(y_test)
    print('       |DONE|')


    nn = NeuralNetwork()
    nn.addLayer(activation=sigmoid, shape=[10, 784])
    nn.epochs = 50
    print('Learning ...')
    nn.fit(X_train, y_train)
    plt.plot(nn.errors)
    plt.show()

    nn.test(X_test, y_test)
    nn.summry()



def encode(y, len=10):
    codes = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    y_encoded = []
    for i in y:
        a = codes[i]
        y_encoded.append(a)

    y_encoded = np.array(y_encoded)
    return y_encoded

def normalize(dataset):
    normized = []
    for matrix in dataset:
        norm = matrix/np.linalg.norm(matrix)
        normized.append(norm)

    norm = np.array(normized)
    return norm


if __name__ == '__main__':
    main()