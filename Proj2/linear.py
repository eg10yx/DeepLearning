import torch
from torch import empty
from module import Module

torch.set_grad_enabled(False)

class Linear(Module):
    def __init__(self, input_size, hidden_unit):
        
        self.input_size = input_size
        self.number_of_hidden_unit = hidden_unit

        self.input = empty(input_size, dtype=torch.float)
        self.output = empty(hidden_unit, dtype=torch.float)

        self.weights = empty(hidden_unit, input_size, dtype=torch.float).uniform_(-1, 1)
        self.biases = empty(hidden_unit, dtype=torch.float).uniform_(-1, 1)

        self.weights_gradients = empty(self.weights.shape, dtype=torch.float).zero_()
        self.biases_gradients = empty(self.biases.shape, dtype=torch.float).zero_()


    def forward(self, input_tensor):

        self.input = input_tensor
        self.output = self.weights @ input_tensor + self.biases

        return self.output

    def backward(self, grad_output):
        
        grad_input = self.weights.transpose(0, 1) @ grad_output
    
        self.biases_gradients = grad_output
        self.weights_gradients = grad_output.view(-1, 1) @ self.input.view(1, -1)
        
        return grad_input

    def gradient_step(self, step_size):

        self.weights -= step_size * self.weights_gradients
        self.biases -= step_size * self.biases_gradients

    def param(self):

        return [(self.weights[i, :], self.weights_gradients[i, :]) for i in range(self.number_of_hidden_unit)] + [(self.biases, self.biases_gradients)]

    def get_number_of_hidden_unit(self):

        return self.number_of_hidden_unit

    def get_input_size(self):

        return self.input_size

    def summary(self):
        print('\tFully connected layer of {} hidden units'.format(self.number_of_hidden_unit))
        