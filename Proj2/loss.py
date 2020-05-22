class MSE:
    def __init__(self):
        pass

    def compute_loss(self, predicted_label, label):

        return ((label-predicted_label)**2).mean(dim=1)

    def compute_grad(self,predicted_label, label):

        return (predicted_label - label).mean(0)