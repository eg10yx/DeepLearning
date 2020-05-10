import torch
from torch import empty
from math import tanh
from module import Module

torch.set_grad_enabled(False)

class Tanh(Module):

    def __init__(self, input_size):
        
        self.hidden_layer_size = input_size
        self.input = empty(input_size)
        self.output = empty(input_size)
        self.grad_input = empty(input_size)

    def forward(self, input_tensor):
        
        self.output = input_tensor.apply_(tanh)
        return self.output

    def backward(self, grad_output):

        diff = 1 - self.output * self.output
        self.grad_input = grad_output * diff
        return self.grad_input

    def gradient_step(self, step_size=None):

        pass

    def param(self):

        return []

    def get_hidden_layer_size(self):

        return self.hidden_layer_size

    def get_input_size(self):
        return self.hidden_layer_size

    def summary(self):
        print('\tTanh activation layer of size {}'.format(self.hidden_layer_size))