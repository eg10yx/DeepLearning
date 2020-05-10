import torch
from torch import empty

torch.set_grad_enabled(False)

class Sequential:
    def __init__(self, loss, input_size):

        self.loss = loss
        self.input_size = input_size
        self.layers = []


    def add_layer(self, layer):
        
        self.layers.append(layer)

    def forward(self, model_input):

        for layer in self.layers:
            model_input = layer.forward(model_input)
        return model_input

    def backward(self, grad_output):

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def gradient_step(self, step_size):

        for layer in self.layers:
            layer.gradient_step(step_size)

    def fit(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=1, step_size=0.1,):

        history = defaultdict(list)

        for epoch in range(1, epochs+1):
            # shuffle indexes in order for GD to look at samples in random order
            idx = list(range(x_train.shape[0]))
            shuffle(idx)

            batches = [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]
            for batch in batches:
                # Forward-pass
                outputs = empty()
                targets = empty()
                for i in batch:
                    output = self.forward(x_train[i])
                    outputs = cat((outputs, output.view(1, -1)), 0)
                    targets = cat((targets, y_train[i].view(1, -1)), 0)

                grad_output = self.loss.compute_grad(outputs, targets)
                self.backward(grad_output)
                self.gradient_step(step_size)

            step_size = step_size * 0.9

            tr_predictions, tr_loss = self.predict(x_train, y_train)
            tr_accuracy = (tr_predictions == y_train).sum() / y_train.shape[1] / tr_predictions.shape[0]
            history['tr_loss'].append(tr_loss.mean())
            history['tr_acc'].append(tr_accuracy)

            val_predictions, val_loss = self.predict(x_test, y_test)
            val_accuracy = (val_predictions == y_test).sum() / y_test.shape[1] / val_predictions.shape[0]
            history['val_loss'].append(val_loss.mean())
            history['val_acc'].append(val_accuracy)
            print('Loss at epoch {} : {}'.format(epoch, tr_loss.mean()))

        print('\nTraining loss : {}'.format(history['tr_loss'][-1]))
        print('Training accuracy : {}\n'.format(history['tr_acc'][-1]))

        print('Test loss : {}'.format(history['val_loss'][-1]))
        print('Test accuracy : {}'.format(history['val_acc'][-1]))

        return history


    def summary(self):
        print('Model with {} layers'.format(len(self.layers)))
        print('\tInput size : {}'.format(self.input_size))
        for layer in self.layers[:-1]:
            layer.summary()
        print('\t{} fully connected output units'.format(self.layers[-1].get_hidden_layer_size()))
