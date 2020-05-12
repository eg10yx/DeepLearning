import math
import torch
from torch import empty
from sequential import Sequential
from linear import Linear
from relu import ReLU
from tanh import Tanh
from losses import MSE


def build_data(n):

    # generate points uniformly at random in [0,1]^2
    coordinates = torch.FloatTensor(n, 2).uniform_(0, 1)
    # create labels (shape (n,)
    labels = ((coordinates - torch.FloatTensor([0.5, 0.5])).norm(p=2, dim=1) < 1 / math.sqrt(2 * math.pi)).type(torch.double)
    # expand labels to one-hot encoding (shape (n,2)
    labels = torch.FloatTensor(n, 2).zero_().scatter_(1, labels.view(-1, 1), 1)
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

history = model.fit(x_train, y_train, x_test, y_test, batch_size=1, epochs=100)