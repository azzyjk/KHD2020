import numpy as np
import random

def setup_data(X, epoch):

    spins = 8000/epoch
    x, y = [], []

    for i in range(spins):
        for j in range(epoch):

            if X[i][j][1] == 1:
                x.append(X[i][j][0])
                y.append(X[i][j][1])
                x.append(X[i][j][0])
                y.append(X[i][j][1])
            else:
                x.append(X[i][j][0])
                y.append(X[i][j][1])

    train_set = []

    for i in range(len(x)):
        train_set.append([x[i], y[i]])

    random.shuffle(train_set)

    x, y = [], []
    for i in range(len(train_set)):
        x.append(train_set[i][0])
        y.append(train_set[i][1])

    x = np.array(x)
    y = np.array(y)

    return x, y
