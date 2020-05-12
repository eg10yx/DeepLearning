from torch import empty

class MSE:
    def __init__(self):
        pass

    def compute_loss(predictions, targets):

        return ((targets-predictions)**2).mean(dim=1)

    def compute_grad(predictions, targets):

        return (predictions - targets).mean(0)