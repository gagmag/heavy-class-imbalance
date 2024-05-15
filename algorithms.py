from training_functions import softmax, hessian_softmax
from networks_here import LogisticRegressionModel

from get_data import get_imbalanced_data

from optimizers import SignSGD, GNSGD, DampedNewton

import torch.optim as optim

def GD_array(config, lr = 1.0, momentum = 0.0):

    model = LogisticRegressionModel(config.d, config.c) # just to remeber c is n in the writeup


    if config.init == "zero":
        model.linear.weight.data.fill_(0)

    #if config.init == "clone":
    #    model.load_state_dict(copy.deepcopy(init_model.state_dict()))
    # random comment

    X, y = get_imbalanced_data(config)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    total_loss_array, group_loss_array, total_acc_array, prob_array \
        = softmax(X, y, model, optimizer, config)

    return total_loss_array, group_loss_array, total_acc_array, prob_array



def SIGN_array(config, lr = 1.0, momentum = 0.0):

    model = LogisticRegressionModel(config.d, config.c) # just to remeber c is n in the writeup


    if config.init == "zero":
        model.linear.weight.data.fill_(0)

    #if config.init == "clone":
    #    model.load_state_dict(copy.deepcopy(init_model.state_dict()))


    X, y = get_imbalanced_data(config)


    optimizer = SignSGD(model.parameters(), lr=lr, momentum=momentum)

    total_loss_array, group_loss_array, total_acc_array, prob_array \
        = softmax(X, y, model, optimizer, config)

    return total_loss_array, group_loss_array, total_acc_array, prob_array



def Adam_array(config, lr = 1.0, momentum = 0.0):

    model = LogisticRegressionModel(config.d, config.c) # just to remeber c is n in the writeup


    if config.init == "zero":
        model.linear.weight.data.fill_(0)

    #if config.init == "clone":
    #    model.load_state_dict(copy.deepcopy(init_model.state_dict()))


    X, y = get_imbalanced_data(config)


    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))

    total_loss_array, group_loss_array, total_acc_array, prob_array \
        = softmax(X, y, model, optimizer, config)

    return total_loss_array, group_loss_array, total_acc_array, prob_array



def RMS_array(config, lr = 1.0, momentum = 0.0):

    model = LogisticRegressionModel(config.d, config.c) # just to remeber c is n in the writeup


    if config.init == "zero":
        model.linear.weight.data.fill_(0)

    #if config.init == "clone":
    #    model.load_state_dict(copy.deepcopy(init_model.state_dict()))


    X, y = get_imbalanced_data(config)


    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    total_loss_array, group_loss_array, total_acc_array, prob_array \
        = softmax(X, y, model, optimizer, config)

    return total_loss_array, group_loss_array, total_acc_array, prob_array



def Newton_array(config, lr = 1.0, damp = 0.0):

    model = LogisticRegressionModel(config.d, config.c) # just to remeber c is n in the writeup


    if config.init == "zero":
        model.linear.weight.data.fill_(0)

    #if config.init == "clone":
    #    model.load_state_dict(copy.deepcopy(init_model.state_dict()))


    X, y = get_imbalanced_data(config)

    total_loss_array, group_loss_array, total_acc_array, prob_array \
        = hessian_softmax(X, y, lr=lr, damp =damp, config=config)

    return total_loss_array, group_loss_array, total_acc_array, prob_array