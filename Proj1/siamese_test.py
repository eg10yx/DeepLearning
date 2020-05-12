import torch
import torch.nn as nn
import torch.nn.functional as F
from dlc_practical_prologue import generate_pair_sets
import matplotlib.pyplot as plt
import numpy as np
import time

# PyTorch printing settings
torch.set_printoptions(threshold=10000)

# PyTorch seeding settings
Manual = False

# Model name
print('Model: Simple CNN\n')

class small_net(nn.Module):
    def __init__(self, siamese=False, auxiliary_loss=False, ch_size=32, k_size=3, nb_conv_layers=2):
        super(small_net, self).__init__()
        # nb_channels: 32-64-128 gives the best result, 16-32-64 is too slow and 64-128-256 overfits the train
        # only two convolutionnal layers give the best result with nb channels: 32-64

        """  Create the correct number of convolutional layers and initialize their weights  """
        

        #self.conv0 = nn.Conv2d(1, 32, 5)
        #nn.init.xavier_normal_(self.conv0.weight)
        #self.conv1 = nn.Conv2d(32, 64, 3)
        #nn.init.xavier_normal_(self.conv1.weight)
        #self.conv2 = nn.Conv2d(64, 128, 3)
        #nn.init.xavier_normal_(self.conv1.weight)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 4, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2)
        )

        """  Initialize linear layers of the network  """

        self.lin0 = nn.Linear(128, 10)
        #nn.init.xavier_normal_(self.lin0.weight)
        self.lin1 = nn.Linear(10, 2)
        #nn.init.xavier_normal_(self.lin1.weight)

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

def compute_nb_errors(data_output, data_target):

    nb_data_errors = 0

    _, predicted_classes = torch.max(data_output, 1)
    for k in range(data_output.size(0)):
        if data_target[k] != predicted_classes[k]:
            nb_data_errors += 1

    return nb_data_errors

def compute_nb_parameters(nb_conv_layers, ch_size, k_size, siamese, auxiliary_loss):

    nb_params = 0
    im_size = 14
    preout_size = 20
    for i in range(nb_conv_layers):

        temp = int((im_size - k_size + 1)/2)
        if temp >= 1:
            im_size = temp
        elif temp >= 0:
            im_size = temp + 1
        else:
            raise ValueError('The specified parameters do not allow to create a correct pre-convolutional network, please revise them.')

        if i == 0:
            if siamese:
                nb_params += int(ch_size*(1*k_size**2 + 1))
            else:
                nb_params += ch_size*(2*k_size**2 + 1)
        else:
            nb_params += ch_size*2**(i)*(ch_size*2**(i-1)*k_size**2 + 1)

    if auxiliary_loss and siamese:
        preout_size = 10   # The preout size is the number of classe for the digits

    nb_params += preout_size*((im_size**2) * ch_size*2**(nb_conv_layers - 1) + 1)
    nb_params += 2*(preout_size + 1)

    return int(nb_params)

# Static hyper-parameters
nb_epochs, nb_folds = 25, 10

# Learning hyper-parameters
lr, batch_size = 0.2, 100

# Network architecture parameters
nb_conv_layers = 2
ch_size = 32
k_size = 3

# Modes
siamese = False
auxiliary_loss = True

"""print('Nb parameters: {}'.format(compute_nb_parameters(nb_conv_layers, ch_size, k_size,
                                                      siamese, auxiliary_loss
                                                      )))"""

# Create tensor to save loss and error along epochs
loss_tensor = np.zeros((nb_epochs, nb_folds))
train_error = np.zeros_like(loss_tensor)
train_aux_error = np.zeros_like(loss_tensor)
test_error = np.zeros_like(loss_tensor)
test_aux_error = np.zeros_like(loss_tensor)

# Data loading
# Set seeding for random data and initializations of network
if Manual:
    torch.random.manual_seed(100)
else:
    torch.random.seed()
nb_data = 1000
train_input = torch.empty(nb_folds, nb_data, 2, 14, 14)
train_target = torch.empty(nb_folds, nb_data, dtype=torch.long)
train_classes = torch.empty(nb_folds, nb_data, 2, dtype=torch.long)
test_input = torch.empty(nb_folds, nb_data, 2, 14, 14)
test_target = torch.empty(nb_folds, nb_data, dtype=torch.long)
test_classes = torch.empty(nb_folds, nb_data, 2, dtype=torch.long)
for i in range(nb_folds):
    train_input[i], train_target[i], train_classes[i], test_input[i], test_target[i], test_classes[i] = generate_pair_sets(nb_data)
    # Normalization step of the data
    normalization = True
    if normalization:
        mu_train, mu_test = train_input[i].mean(0), test_input[i].mean(0)

        train_input[i].sub_(mu_train).div_(255)
        test_input[i].sub_(mu_test).div_(255)

################## TRAINING ##################

start = time.time()

for f in range(nb_folds):

    print('Fold {}'.format(f+1))

    # Set seeding for random data and initializations of network
    if Manual:
        torch.random.manual_seed(f)
    else:
        torch.random.seed()

    #Choice of dataset for this fold
    fold_train_input = train_input[f]
    fold_train_target = train_target[f]
    fold_train_classes = train_classes[f]
    fold_test_input = test_input[f]
    fold_test_target = test_target[f]
    fold_test_classes = test_classes[f]

    # Model creation and choices of Loss and Optimizer
    model = small_net(siamese, auxiliary_loss, ch_size, k_size, nb_conv_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for e in range(nb_epochs):

        # Set the model to training mode
        model.train()

        for input, target, classes in zip(fold_train_input.split(batch_size), fold_train_target.split(batch_size), fold_train_classes.split(batch_size)):

            # Compute model output
            in1 = input[:, 0, :, :].reshape(input.size(0), 1, 14, 14)
            in2 = input[:, 1, :, :].reshape(input.size(0), 1, 14, 14)
            preout1, preout2, output = model(in1, in2)
            
            # Train the model
            loss = criterion(output, target)
            if auxiliary_loss:
                aux_loss = criterion(preout1, classes[:, 0]) + criterion(preout2, classes[:, 1])
                loss += aux_loss
            loss_tensor[e, f] +=  loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute current epoch's training error
            train_error[e, f] += compute_nb_errors(output, target)*100/nb_data

            if auxiliary_loss:
                train_aux_error[e, f] += compute_nb_errors(preout1, classes[:, 0])*100/(2*nb_data)
                train_aux_error[e, f] += compute_nb_errors(preout2, classes[:, 1])*100/(2*nb_data)


        # Set the model to testing mode to compute test error at current epoch
        model.eval()
        test_in1 = fold_test_input[:, 0, :, :].reshape(fold_test_input.size(0), 1, 14, 14)
        test_in2 = fold_test_input[:, 1, :, :].reshape(fold_test_input.size(0), 1, 14, 14)

        test_preout1, test_preout2, test_output = model(test_in1, test_in2)
        test_error[e, f] = compute_nb_errors(test_output, fold_test_target)*100/nb_data

        if auxiliary_loss:
            test_aux_error[e, f] += compute_nb_errors(test_preout1, fold_test_classes[:, 0])*100/(2*nb_data)
            test_aux_error[e, f] += compute_nb_errors(test_preout2, fold_test_classes[:, 1])*100/(2*nb_data)

#Time
print('Elapsed time: {}s'.format(time.time()-start))

# Compute mean and std of the errors over the folds
train_err_mean, train_err_std = train_error.mean(axis=1), train_error.std(axis=1)
test_err_mean, test_err_std = test_error.mean(axis=1), test_error.std(axis=1)
train_aux_err_mean, train_aux_err_std = train_aux_error.mean(axis=1), train_aux_error.std(axis=1)
test_aux_err_mean, test_aux_err_std = test_aux_error.mean(axis=1), test_aux_error.std(axis=1)


# Print train error on the whole dataset and test error
print('Average train error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, train_err_mean[-1], train_err_std[-1]))
print('Average test error over {} folds: mean = {:.1f}%, std = {:.2f}\n'.format(nb_folds ,test_err_mean[-1], test_err_std[-1]))
if auxiliary_loss:
    print('Average auxiliary train error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, train_aux_err_mean[-1], train_aux_err_std[-1]))
    print('Average auxiliary test error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, test_aux_err_mean[-1], test_aux_err_std[-1]))

# Plot the evolution of the train and test final error with respect to the epoch
plt.figure('Average train and test final error over {} folds'.format(nb_folds))
plt.errorbar(np.arange(1, nb_epochs+1, 1), train_err_mean, yerr=train_err_std, linewidth=2.0, 
            label='Train', elinewidth=1.0, lolims=True, uplims=True, errorevery=(4,5)
            )
plt.errorbar(np.arange(1, nb_epochs+1, 1), test_err_mean, yerr=test_err_std, linewidth=2.0,
            label='Test', elinewidth=1.0, lolims=True, uplims=True, errorevery=(4,5)
            )
plt.title('Evolution over {} folds of average final train and test error'.format(nb_folds))
plt.xlabel('Epochs')
plt.ylabel('Error in %')
plt.legend()

# Plot the evolution of the train and test auxiliary error with respect to the epoch
if auxiliary_loss:
    plt.figure('Average train and test auxiliary error over {} folds'.format(nb_folds))
    plt.errorbar(np.arange(1, nb_epochs+1, 1), train_aux_err_mean, yerr=train_aux_err_std, linewidth=2.0, 
                label='Train', elinewidth=1.0, lolims=True, uplims=True, errorevery=(4,5)
                )
    plt.errorbar(np.arange(1, nb_epochs+1, 1), test_aux_err_mean, yerr=test_aux_err_std, linewidth=2.0,
                label='Test', elinewidth=1.0, lolims=True, uplims=True, errorevery=(4,5)
                )
    plt.title('Evolution over {} folds of average auxiliary train and test error'.format(nb_folds))
    plt.xlabel('Epochs')
    plt.ylabel('Error in %')
    plt.legend()


plt.show()