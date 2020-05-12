class trackhistory():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.accuracy = []
        self.loss = []
        self.val_accuracy = []
        self.val_loss = []


    def on_epoch_end(self, epoch):

        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        loss, acc = self.evaluate(x_train, y_train)
        val_loss, val_acc = self.evaluate(x_test, y_test)

        self.loss.append(loss)
        self.accuracy.append(acc)
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_acc)
        print('\ntrain loss: {}, train acc: {}'.format(loss, acc))
        print('\nval loss: {}, val acc: {}\n\n'.format(val_loss, val_acc))
        
    def evaluate(self, x, y):

        output_size = self.layers[-1].get_hidden_size()
        predictions = FloatTensor(x.shape[0], output_size).zero_()

        for i in range(x.shape[0]):
            predictions[i] = self.forward(x[i])

        loss = self.loss.compute_loss(predictions, y)
        _, ind = predictions.max(1)
        predictions = FloatTensor(predictions.shape).zero_().scatter_(1, ind.view(-1, 1), 1)
        return loss
