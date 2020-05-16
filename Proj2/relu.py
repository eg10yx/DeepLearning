import torch
from torch import empty
from module import Module

torch.set_grad_enabled(False)

class ReLU(Module):

    def __init__(self, input_size):

        self.hidden_layer_size = input_size
        self.input = empty(input_size, dtype=torch.float)
        self.output = empty(input_size, dtype=torch.float)
        self.grad_input = empty(input_size, dtype=torch.float)

    def forward(self, input_tensor):

        self.output = input_tensor
        self.output[input_tensor < 0] = 0
        return self.output

    def backward(self, grad_output):

        diff = self.output
        diff[diff != 0] = 1
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
        print('\tReLU activation layer of size {}'.format(self.hidden_layer_size))