import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from dlc_practical_prologue import generate_pair_sets
from models_A import A_basic, A_siamese, A_auxiliary_loss, A_siamese_and_auxiliary_loss
from models_B import B_basic, B_siamese, B_auxiliary_loss, B_siamese_and_auxiliary_loss
from useful_func import compute_nb_errors
from useful_func import plot_errorbar

# PyTorch seeding settings
Manual = False

# CNN types settings
siamese_auxiliary_loss = [(False, False), (True, False), (False, True), (True, True)]

# Static hyper-parameters
nb_epochs, nb_folds, batch_size = 25, 10, 100



"""  -----------------------------------------------------------------------  """
############################### TRAINING MODELS A ###############################
"""  -----------------------------------------------------------------------  """


# Time
start_time = time.time()
training_time = 0

# Leanring hyper-parameter
lr = 0.15

for cnn_index, cnn_type in enumerate(siamese_auxiliary_loss):

    siam, aux = cnn_type

    # Create tensor to save loss and error along epochs
    loss_tensor = np.zeros((nb_epochs, nb_folds))
    train_error = np.zeros_like(loss_tensor)
    train_aux_error = np.zeros_like(loss_tensor)
    test_error = np.zeros_like(loss_tensor)
    test_aux_error = np.zeros_like(loss_tensor)

    print('Start training over {} folds for CNN-A type siamese = {}, auxiliary_loss = {}.\n'.format(nb_folds, siam, aux))

    for f in range(nb_folds):

        print('Fold {}/{}...'.format(f+1, nb_folds))

        # Set seeding for random data and initializations of network
        if Manual:
            torch.random.manual_seed(f)
        else:
            torch.random.seed()

        # Data loading
        nb_data = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb_data)

        # Normalization step of the data
        normalization = True
        if normalization:
            mu_train, mu_test = train_input.mean(0), test_input.mean(0)

            train_input.sub_(mu_train).div_(255)
            test_input.sub_(mu_test).div_(255)

        # Model creation and choices of Loss and Optimizer
        if siam:
            if aux:
                model = A_siamese_and_auxiliary_loss()
            else:
                model = A_siamese()
        else:
            if aux:
                model = A_auxiliary_loss()
            else:
                model = A_basic()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for e in range(nb_epochs):

            # Set the model to training mode
            model.train()

            for input, target, classes in zip(train_input.split(batch_size), train_target.split(batch_size), train_classes.split(batch_size)):
                
                if siam:
                    with torch.no_grad():
                        in1 = input[:, 0, :, :].reshape(input.size(0), 1, 14, 14)
                        in2 = input[:, 1, :, :].reshape(input.size(0), 1, 14, 14)
                    if aux:
                        preout1, preout2, output = model(in1, in2)
                    else:
                        output = model(in1, in2)
                else:
                    if aux:
                        preout, output = model(input)
                    else:
                        output = model(input)
                
                # Train the model
                loss = criterion(output, target)
                if aux:
                    if siam:
                        aux_loss = criterion(preout1, classes[:, 0]) + criterion(preout2, classes[:, 1])
                    else:
                        aux_loss = criterion(preout[:, :10], classes[:, 0]) + criterion(preout[:, 10:], classes[:, 1])
                    loss += aux_loss
                loss_tensor[e, f] +=  loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute current epoch's training error
                train_error[e, f] += compute_nb_errors(output, target)*100/nb_data

                if aux:
                    if siam:
                        train_aux_error[e, f] += compute_nb_errors(preout1, classes[:, 0])*100/(2*nb_data)
                        train_aux_error[e, f] += compute_nb_errors(preout2, classes[:, 1])*100/(2*nb_data)
                    else:
                        train_aux_error[e, f] += compute_nb_errors(preout[:, :10], classes[:, 0])*100/(2*nb_data)
                        train_aux_error[e, f] += compute_nb_errors(preout[:, 10:], classes[:, 1])*100/(2*nb_data)


            # Set the model to testing mode to compute test error at current epoch
            model.eval()

            if siam:
                with torch.no_grad():
                    test_in1 = test_input[:, 0, :, :].reshape(test_input.size(0), 1, 14, 14)
                    test_in2 = test_input[:, 1, :, :].reshape(test_input.size(0), 1, 14, 14)
                if aux:
                    test_preout1, test_preout2, test_output = model(test_in1, test_in2)
                else:
                    test_output = model(test_in1, test_in2)
            else:
                if aux:
                    test_preout, test_output = model(test_input)
                else:
                    test_output = model(test_input)
            
            test_error[e, f] = compute_nb_errors(test_output, test_target)*100/nb_data

            if aux:
                if siam:
                    test_aux_error[e, f] += compute_nb_errors(test_preout1, test_classes[:, 0])*100/(2*nb_data)
                    test_aux_error[e, f] += compute_nb_errors(test_preout2, test_classes[:, 1])*100/(2*nb_data)
                else:
                    test_aux_error[e, f] += compute_nb_errors(test_preout[:, :10], test_classes[:, 0])*100/(2*nb_data)
                    test_aux_error[e, f] += compute_nb_errors(test_preout[:, 10:], test_classes[:, 1])*100/(2*nb_data)
    
    # Time
    training_time += time.time() - start_time
    
    # Compute mean and std of the errors over the folds
    loss_mean, loss_std = loss_tensor.mean(axis=1), loss_tensor.std(axis=1)
    train_err_mean, train_err_std = train_error.mean(axis=1), train_error.std(axis=1)
    test_err_mean, test_err_std = test_error.mean(axis=1), test_error.std(axis=1)
    train_aux_err_mean, train_aux_err_std = train_aux_error.mean(axis=1), train_aux_error.std(axis=1)
    test_aux_err_mean, test_aux_err_std = test_aux_error.mean(axis=1), test_aux_error.std(axis=1)

    # Print train error on the whole dataset and test error and loss
    print('\nAverage loss over {} folds: mean = {:.1f}, std = {:.2f}\n'.format(nb_folds, loss_mean[-1], loss_std[-1]))

    print('Average train error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, train_err_mean[-1], train_err_std[-1]))
    print('Average test error over {} folds: mean = {:.1f}%, std = {:.2f}\n'.format(nb_folds ,test_err_mean[-1], test_err_std[-1]))
    if aux:
        print('Average auxiliary train error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, train_aux_err_mean[-1], train_aux_err_std[-1]))
        print('Average auxiliary test error over {} folds: mean = {:.1f}%, std = {:.2f}\n'.format(nb_folds, test_aux_err_mean[-1], test_aux_err_std[-1]))
    print('\n')

    # Plot the evolution of the train and test final error with respect to the epoch
    plt.figure('A-Average train and test final error over {} folds, siamese = {}, auxiliary_loss = {}'.format(nb_folds, siam, aux))

    plot_errorbar(test_err_mean, test_err_std, nb_folds, nb_epochs, siam, aux, train_err_mean=train_err_mean, train_err_std=train_err_std)

    # Plot the evolution of the train and test auxiliary error with respect to the epoch
    if aux:
        plt.figure('A-Average train and test auxiliary error over {} folds, siamese = {}, auxiliary_loss = {}'.format(nb_folds, siam, aux))

        plot_errorbar(test_aux_err_mean, test_aux_err_std, nb_folds, nb_epochs, siam, aux, train_err_mean=train_aux_err_mean, train_err_std=train_aux_err_std)

    # Plot the test error evolution of the different models
    plt.figure('A-Test error for different models')

    plot_errorbar(test_err_mean, test_err_std, nb_folds, nb_epochs, siam, aux, index=cnn_index)

    # Plot the loss evolution of the different models
    plt.figure('A-Loss')

    plot_errorbar(loss_mean, loss_std, nb_folds, nb_epochs, siam, aux, index=cnn_index+4)

    start_time = time.time()

training_time_A = training_time
print('Total elapsed time for trainign CNN-As: {}s.\n\n'.format(training_time_A))

print('-------------------------------------------------------\n\n')

"""  -----------------------------------------------------------------------  """
############################### TRAINING MODELS B ###############################
"""  -----------------------------------------------------------------------  """

# Leanring hyper-parameter
lr = 0.015

# Time
start_time = time.time()

for cnn_index, cnn_type in enumerate(siamese_auxiliary_loss):

    siam, aux = cnn_type

    # Create tensor to save loss and error along epochs
    loss_tensor = np.zeros((nb_epochs, nb_folds))
    train_error = np.zeros_like(loss_tensor)
    train_aux_error = np.zeros_like(loss_tensor)
    test_error = np.zeros_like(loss_tensor)
    test_aux_error = np.zeros_like(loss_tensor)

    print('Start training over {} folds for CNN-B type siamese = {}, auxiliary_loss = {}.\n'.format(nb_folds, siam, aux))

    for f in range(nb_folds):

        print('Fold {}/{}...'.format(f+1, nb_folds))

        # Set seeding for random data and initializations of network
        if Manual:
            torch.random.manual_seed(f)
        else:
            torch.random.seed()

        # Data loading
        nb_data = 1000
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb_data)

        # Normalization step of the data
        normalization = True
        if normalization:
            mu_train, mu_test = train_input.mean(0), test_input.mean(0)

            train_input.sub_(mu_train).div_(255)
            test_input.sub_(mu_test).div_(255)

        # Model creation and choices of Loss and Optimizer
        if siam:
            if aux:
                model = B_siamese_and_auxiliary_loss()
            else:
                model = B_siamese()
        else:
            if aux:
                model = B_auxiliary_loss()
            else:
                model = B_basic()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for e in range(nb_epochs):

            # Set the model to training mode
            model.train()

            for input, target, classes in zip(train_input.split(batch_size), train_target.split(batch_size), train_classes.split(batch_size)):
                
                if siam:
                    with torch.no_grad():
                        in1 = input[:, 0, :, :].reshape(input.size(0), 1, 14, 14)
                        in2 = input[:, 1, :, :].reshape(input.size(0), 1, 14, 14)
                    if aux:
                        preout1, preout2, output = model(in1, in2)
                    else:
                        output = model(in1, in2)
                else:
                    if aux:
                        preout, output = model(input)
                    else:
                        output = model(input)
                
                # Train the model
                loss = criterion(output, target)
                if aux:
                    if siam:
                        aux_loss = criterion(preout1, classes[:, 0]) + criterion(preout2, classes[:, 1])
                    else:
                        aux_loss = criterion(preout[:, :10], classes[:, 0]) + criterion(preout[:, 10:], classes[:, 1])
                    loss += aux_loss
                loss_tensor[e, f] +=  loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute current epoch's training error
                train_error[e, f] += compute_nb_errors(output, target)*100/nb_data

                if aux:
                    if siam:
                        train_aux_error[e, f] += compute_nb_errors(preout1, classes[:, 0])*100/(2*nb_data)
                        train_aux_error[e, f] += compute_nb_errors(preout2, classes[:, 1])*100/(2*nb_data)
                    else:
                        train_aux_error[e, f] += compute_nb_errors(preout[:, :10], classes[:, 0])*100/(2*nb_data)
                        train_aux_error[e, f] += compute_nb_errors(preout[:, 10:], classes[:, 1])*100/(2*nb_data)


            # Set the model to testing mode to compute test error at current epoch
            model.eval()

            if siam:
                with torch.no_grad():
                    test_in1 = test_input[:, 0, :, :].reshape(test_input.size(0), 1, 14, 14)
                    test_in2 = test_input[:, 1, :, :].reshape(test_input.size(0), 1, 14, 14)
                if aux:
                    test_preout1, test_preout2, test_output = model(test_in1, test_in2)
                else:
                    test_output = model(test_in1, test_in2)
            else:
                if aux:
                    test_preout, test_output = model(test_input)
                else:
                    test_output = model(test_input)
            
            test_error[e, f] = compute_nb_errors(test_output, test_target)*100/nb_data

            if aux:
                if siam:
                    test_aux_error[e, f] += compute_nb_errors(test_preout1, test_classes[:, 0])*100/(2*nb_data)
                    test_aux_error[e, f] += compute_nb_errors(test_preout2, test_classes[:, 1])*100/(2*nb_data)
                else:
                    test_aux_error[e, f] += compute_nb_errors(test_preout[:, :10], test_classes[:, 0])*100/(2*nb_data)
                    test_aux_error[e, f] += compute_nb_errors(test_preout[:, 10:], test_classes[:, 1])*100/(2*nb_data)
    
    # Time
    training_time += time.time() - start_time
    
    # Compute mean and std of the errors over the folds
    loss_mean, loss_std = loss_tensor.mean(axis=1), loss_tensor.std(axis=1)
    train_err_mean, train_err_std = train_error.mean(axis=1), train_error.std(axis=1)
    test_err_mean, test_err_std = test_error.mean(axis=1), test_error.std(axis=1)
    train_aux_err_mean, train_aux_err_std = train_aux_error.mean(axis=1), train_aux_error.std(axis=1)
    test_aux_err_mean, test_aux_err_std = test_aux_error.mean(axis=1), test_aux_error.std(axis=1)

    # Print train error on the whole dataset and test error and loss
    print('\nAverage loss over {} folds: mean = {:.1f}, std = {:.2f}\n'.format(nb_folds, loss_mean[-1], loss_std[-1]))

    print('Average train error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, train_err_mean[-1], train_err_std[-1]))
    print('Average test error over {} folds: mean = {:.1f}%, std = {:.2f}\n'.format(nb_folds ,test_err_mean[-1], test_err_std[-1]))
    if aux:
        print('Average auxiliary train error over {} folds: mean = {:.1f}%, std = {:.2f}'.format(nb_folds, train_aux_err_mean[-1], train_aux_err_std[-1]))
        print('Average auxiliary test error over {} folds: mean = {:.1f}%, std = {:.2f}\n'.format(nb_folds, test_aux_err_mean[-1], test_aux_err_std[-1]))
    print('\n')

    # Plot the evolution of the train and test final error with respect to the epoch
    plt.figure('B-Average train and test final error over {} folds, siamese = {}, auxiliary_loss = {}'.format(nb_folds, siam, aux))

    plot_errorbar(test_err_mean, test_err_std, nb_folds, nb_epochs, siam, aux, train_err_mean=train_err_mean, train_err_std=train_err_std)

    # Plot the evolution of the train and test auxiliary error with respect to the epoch
    if aux:
        plt.figure('B-Average train and test auxiliary error over {} folds, siamese = {}, auxiliary_loss = {}'.format(nb_folds, siam, aux))

        plot_errorbar(test_aux_err_mean, test_aux_err_std, nb_folds, nb_epochs, siam, aux, train_err_mean=train_aux_err_mean, train_err_std=train_aux_err_std)

    # Plot the test error evolution of the different models
    plt.figure('B-Test error for different models')

    plot_errorbar(test_err_mean, test_err_std, nb_folds, nb_epochs, siam, aux, index=cnn_index)

    # Plot the loss evolution of the different models
    plt.figure('B-Loss')

    plot_errorbar(loss_mean, loss_std, nb_folds, nb_epochs, siam, aux, index=cnn_index+4)

    start_time = time.time()

# Time
training_time_B = training_time - training_time_A
print('Total elapsed time for trainign CNN-Bs: {}s.\n'.format(training_time_B))
print('Total elapsed time: {}s\n'.format(training_time))

# Show plots
plt.show()