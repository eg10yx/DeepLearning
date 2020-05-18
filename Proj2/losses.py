class MSE:
    def __init__(self):
        pass

    def compute_loss(self, predicted_class, targets):

        return ((targets-predicted_class)**2).mean(dim=1)

    def compute_grad(self,predicted_class, targets):

        return (predicted_class - targets).mean(0)