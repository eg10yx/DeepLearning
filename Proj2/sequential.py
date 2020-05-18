import torch
from torch import empty
from random import shuffle
from torch import cat
torch.set_grad_enabled(False)
class Sequential:
    def __init__(self, loss, input_size):

        self.loss = loss
        self.input_size = input_size
        self.layers = []


    def add_layer(self, layer):
        
        self.layers.append(layer)

    def forward(self, layer_input):

        for layer in self.layers:
            layer_input = layer.forward(layer_input)
            
        return layer_input

    def backward(self, grad_output):

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def gradient_step(self, step_size):

        for layer in self.layers:
            layer.gradient_step(step_size)

    def fit(self, x_train, y_train, x_test, y_test, epochs=100, step_size=0.1):
        
        history = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])

        for epoch in range(1, epochs+1):
            print('\nepoch number: {}'.format(epoch))
            
            idx = torch.arange(x_train.shape[0])
            shuffle(idx)
            
            outputs = empty(0, dtype=torch.float)
            targets = empty(0, dtype=torch.float)
            
            for i in idx:
                output = self.forward(x_train[i])
                
                outputs = cat((outputs, output.view(1, -1)), 0)
                targets = cat((targets, y_train[i].view(1, -1)), 0)

            grad_output = self.loss.compute_grad(output, targets)
            
            self.backward(grad_output)
            self.gradient_step(step_size)
            
            step_size = step_size * 0.9
                
            history = self.evaluate(x_train, y_train, history, 'train')
            history = self.evaluate(x_test, y_test, history, 'test')
        
        return history


    def evaluate(self, x,  y, history, split):
        
        output_size = self.layers[-1].get_hidden_layer_size()
        predictions = empty(x.shape[0], output_size, dtype=torch.float).zero_()
        
        
        for i in range(x.shape[0]):
            predictions[i] = self.forward(x[i])
            
        error = predictions.max(1)[1].ne(y.max(1)[1]).sum(dtype=torch.float)/predictions.size(0)
        print("Error: {:6.2%}\n".format(error))

        loss = self.loss.compute_loss(predictions, y)

        _, ind = predictions.max(1)
        predictions = empty(predictions.shape, dtype=torch.float).zero_().scatter_(1, ind.view(-1, 1), 1)
        accuracy = (predictions == y).sum(dtype=torch.float) / y.shape[1] / predictions.shape[0]

        if split == 'train':
            history['train_loss'].append(loss.mean())
            history['train_acc'].append(accuracy)
            print('\ntrain loss: {}, train accuracy: {}'.format(loss.mean(), accuracy))
        else:
            history['test_loss'].append(loss.mean())
            history['test_acc'].append(accuracy)
            print('\ntest loss: {}, test accuracy: {}\n\n'.format(loss.mean(), accuracy))
        
        return history
            
        
    def summary(self):
        print('Model with {} layers'.format(len(self.layers)))
        print('\tInput size : {}'.format(self.input_size))
        for layer in self.layers[:-1]:
            layer.summary()
        print('\t{} fully connected output units'.format(self.layers[-1].get_hidden_layer_size()))
