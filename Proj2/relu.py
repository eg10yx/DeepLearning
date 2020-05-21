import torch
from torch import empty
from module import Module

torch.set_grad_enabled(False)

class ReLU(Module):

    def __init__(self, input_size):

        self.number_of_hidden_unit = input_size
        self.input = empty(input_size, dtype=torch.float)
        self.output = empty(input_size, dtype=torch.float)
        self.grad_input = empty(input_size, dtype=torch.float)

    def forward(self, input_tensor):

        self.output = input_tensor
        self.output =torch.clamp(self.output, min=0)
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

    def get_number_of_hidden_unit(self):

        return self.number_of_hidden_unit

    def get_input_size(self):

        return self.number_of_hidden_unit

    def summary(self):
        print('\tReLU activation layer of size {}'.format(self.number_of_hidden_unit))