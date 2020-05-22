import torch
import matplotlib.pyplot as plt
import numpy as np

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

def plot_errorbar(test_err_mean, test_err_std, nb_folds, nb_epochs, siamese, aux_loss, train_err_mean=None, train_err_std=None, index=None):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    labels = ['Basic CNN', 'Siamese CNN', 'CNN with aux. loss', 'Siamese CNN with aux. loss']

    if np.any(train_err_mean):
        plt.plot(np.arange(1, 6, 1), train_err_mean[:5], linewidth=2.0, color=colors[0])
        plt.errorbar(np.arange(5, nb_epochs+1, 1), train_err_mean[4:], yerr=train_err_std[4:], linewidth=2.0, 
                    label='Train', elinewidth=1.0, lolims=True, uplims=True, errorevery=5, color=colors[0]
                    )
    
    if index or index == 0:
        color = colors[index%4]
        label = labels[index%4]
    else:
        color = colors[1]
        label = 'Test'

    plt.plot(np.arange(1, 6, 1), test_err_mean[:5], linewidth=2.0, color=color)
    plt.errorbar(np.arange(5, nb_epochs+1, 1), test_err_mean[4:], yerr=test_err_std[4:], linewidth=2.0,
                label=label, elinewidth=1.0, lolims=True, uplims=True, errorevery=5, color=color
                )

    plt.xlabel('Epochs')
    if index and index > 3:
        plt.ylabel('Loss')
    else:
        plt.ylabel('Error in %')
    plt.legend()

    pass