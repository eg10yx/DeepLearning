import torch
import torch.nn as nn
import torch.nn.functional as F

class B_basic(nn.Module):
    """
    Basic CNN model with batch norm:

    Input goes through a conventional convolutional network of three layers before going
    through a fully connected linear layer, a ReLU and another fully connected layer.
    """

    def __init__(self):
        super(B_basic, self).__init__()
        """  Create the correct number of convolutional layers and initialize their weights  """
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 4, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128)
        )

        """  Initialize linear layers of the network  """

        self.lin0 = nn.Linear(128, 10)
        
        self.lin1 = nn.Linear(10, 2)
        

    def pre_forward(self, x):

        x = self.conv(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin0(x))

        return x

    def forward(self, x):

        preout = self.pre_forward(x)

        out = self.lin1(preout)

        return out

class B_siamese(nn.Module):
    """
    Siamese CNN model:

    This network use two inputs. Each of them goes through a conventional convolutional network of three layers before going
    through a fully connected linear layer and a ReLU. Then these partly processed inputs are subtracted from one another and
    this difference goes through a final fully connected layer.
    """

    def __init__(self):
        super(B_siamese, self).__init__()
        """  Create the correct number of convolutional layers and initialize their weights  """
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 4, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128)
        )

        """  Initialize linear layers of the network  """

        self.lin0 = nn.Linear(128, 10)
        
        self.lin1 = nn.Linear(10, 2)
        

    def pre_forward(self, x):

        x = self.conv(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin0(x))

        return x

    def forward(self, x1, x2):

        preout1 = self.pre_forward(x1)
        preout2 = self.pre_forward(x2)

        dif = preout1 - preout2

        out = self.lin1(dif)

        return out

class B_auxiliary_loss(nn.Module):
    """
    Auxiliary loss CNN model:

    This network use one input. A preoutput is computed to use it for an auxiliary loss. This preoutput of size Mx20 is
    supposed to converge towards the class of the two digits given (the 10 first elements are used for the first digit in
    the training and the 10 last for the second one). The preoutput is computed by processing the input in a conventional
    convolutional network followed by a fully connected layer and a ReLU. Then the final output is computed by going through
    a final fully connected layer.
    """

    def __init__(self):
        super(B_auxiliary_loss, self).__init__()
        """  Create the correct number of convolutional layers and initialize their weights  """
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 4, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128)
        )

        """  Initialize linear layers of the network  """

        self.lin0 = nn.Linear(128, 20)
        
        self.lin1 = nn.Linear(20, 2)
        

    def pre_forward(self, x):

        x = self.conv(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin0(x))

        return x

    def forward(self, x):

        preout = self.pre_forward(x)

        out = self.lin1(preout)

        return preout, out

class B_siamese_and_auxiliary_loss(nn.Module):
    """
    Siamese and auxiliary loss CNN model:

    This network use two inputs, two digits. For each input, a preoutput is computed and returned to be used for an
    auxiliary loss. This is computed by going through a conventional convolutional network followed by a fully connected
    layer and a ReLU. Those two preoutputs are substracted from one another and this difference go through a final 
    fully connected layer to compute the final output.
    """

    def __init__(self):
        super(B_siamese_and_auxiliary_loss, self).__init__()
        """  Create the correct number of convolutional layers and initialize their weights  """
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 4, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128)
        )

        """  Initialize linear layers of the network  """

        self.lin0 = nn.Linear(128, 10)
        
        self.lin1 = nn.Linear(10, 2)
        

    def pre_forward(self, x):

        x = self.conv(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin0(x))

        return x

    def forward(self, x1, x2):

        preout1 = self.pre_forward(x1)
        preout2 = self.pre_forward(x2)

        dif = preout1 - preout2

        out = self.lin1(dif)

        return preout1, preout2, out