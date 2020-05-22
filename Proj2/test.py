import math
import torch
import matplotlib.pyplot as plt
from torch import empty
from sequential import Sequential
from linear import Linear
from relu import ReLU
from tanh import Tanh
from losses import MSE
from plot_functions import plot_history


def build_data(n):

    coordinates = empty(n, 2, dtype=torch.float).uniform_(0, 1)
    labels = ((coordinates - torch.Tensor([0.5, 0.5])).norm(p=2, dim=1) < 1 / math.sqrt(2 * math.pi)).type(torch.LongTensor)
    labels = empty(n, 2, dtype=torch.float).zero_().scatter_(1, labels.view(-1, 1), 1)
    return coordinates, labels


def build_model():
    model = Sequential(MSE(), input_size=2)
    model.add_layer(Linear(2, 25))
    model.add_layer(ReLU(25))
    model.add_layer(Linear(25, 25))
    model.add_layer(ReLU(25))
    model.add_layer(Linear(25, 25))
    model.add_layer(Tanh(25))
    model.add_layer(Linear(25, 2))
    return model


x_train, y_train = build_data(1000)
x_test, y_test = build_data(1000)

model = build_model()

model.summary()

history = model.fit(x_train, y_train, x_test, y_test, epochs=60)

plot_history(history, 'Network with two input units, two output units, three hidden layers of 25 units')
plt.show()