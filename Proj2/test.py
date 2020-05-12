import math
import torch
from torch import empty
from sequential import Sequential
from linear import Linear
from relu import ReLU
from tanh import Tanh
from losses import MSE


def build_data(n):

    x = empty(n, 2, dtype=torch.double).uniform_(-0.5,0.5)
    y = x.pow(2).sum(1).sub(1 /(2* math.pi)).sign().mul(-1).add(1).div(2).double()
    return x, y


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

history = model.fit(x_train, y_train, x_test, y_test, batch_size=5, epochs=100)

