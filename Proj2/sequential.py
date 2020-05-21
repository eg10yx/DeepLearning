import torch
from torch import empty
from random import shuffle

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

    def fit(self, x_train, y_train, x_test, y_test, epochs=100, step_size=0.02):
        """
        Train and test the model for each epoch.
        The reducing step size help to converge more precisely
        
        """
        history = dict(train_loss=[], test_loss=[], train_error=[], test_error=[])
        
        for epoch in range(1, epochs+1):
            print('\nepoch n°: {}'.format(epoch))
            
            idx = torch.arange(x_train.shape[0])
            shuffle(idx)
            
            output = empty((1,2), dtype=torch.float)
            target = empty((1,2), dtype=torch.float)
            
            for i in idx:

                output[0] = self.forward(x_train[i])
                target[0] = y_train[i]

                grad_output = self.loss.compute_grad(output, target)
                
                self.backward(grad_output)

                self.gradient_step(step_size)

            step_size = step_size * 0.85
            
            history = self.evaluate(x_train, y_train, history, 'train')
            history = self.evaluate(x_test, y_test, history, 'test')
        
        return history


    def evaluate(self, x,  y, history, split):
        """
        Compute Loss and error and add them to the history dictionary for further plot
        
        """
        output_size = self.layers[-1].get_number_of_hidden_unit()
        predicted_class = empty(x.shape[0], output_size, dtype=torch.float).zero_()
        
        for i in range(x.shape[0]):
            predicted_class[i] = self.forward(x[i])
            
        error = predicted_class.max(1)[1].ne(y.max(1)[1]).sum(dtype=torch.float)/predicted_class.size(0)

        loss = self.loss.compute_loss(predicted_class, y)

        if split == 'train':
            history['train_loss'].append(loss.mean())
            history['train_error'].append(error*100)
            print('\ntrain loss: {}, train error: {:6.2%}\n'.format(loss.mean(), error))
            
        elif split == 'test':
            history['test_loss'].append(loss.mean())
            history['test_error'].append(error*100)
            print('\ntest loss: {}, test error: {:6.2%}\n\n'.format(loss.mean(), error))
        
        return history
            
        
    def summary(self):
        print('Model with {} layers'.format(len(self.layers)))
        print('\tInput size : {}'.format(self.input_size))
        for layer in self.layers[:-1]:
            layer.summary()
        print('\t{} fully connected output units'.format(self.layers[-1].get_number_of_hidden_unit()))
