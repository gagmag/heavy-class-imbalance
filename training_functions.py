import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def softmax(X, y, model, optimizer, config):



    criterion = nn.CrossEntropyLoss(reduction = 'none')
    
    #criterion = nn.BCEWithLogitsLoss(reduction = 'none')

    if config.group_size is not None:
        group_loss_array = [[] for i in range(config.group_size)] # will track the group losses


    total_loss_array = [] # this is for tracking the total loss (it's the sum of group 0 and group 1)
    prob_array = [] # this is for just tracking some probability values
    total_acc_array = [] # this is for tracking the total accuracy

    for epoch in tqdm.tqdm(range(config.num_epochs)):


        optimizer.zero_grad()

        output = model(X)
        
        pred = torch.argmax(output, dim=1)

        prob_array.append(output) # just storing the unnormalized probs for some calculations later

        loss_val_array = criterion(output, y)
        loss = loss_val_array.mean()

        loss.backward(create_graph=True)
        optimizer.step()


        total_loss_array.append(loss.item()) # the total loss



        with torch.no_grad():

            total_acc_array.append((pred == y).float().mean().item())

            if config.group_size is not None:

                for i in range(config.group_size):
                    group_loss_array[i].append(0)

                for i in range(loss_val_array.shape[0]):

                    group_num = config.group_array[y[i].item()]
                    group_loss_array[group_num][-1] += loss_val_array[i].item()


    return total_loss_array, group_loss_array, total_acc_array, prob_array


from torch.func import jacfwd
from functorch import hessian

def hessian_softmax(X, y, lr, damp, config):



    criterion = nn.CrossEntropyLoss(reduction = 'none')
    param = torch.zeros(config.d * config.c)


    if config.group_size is not None:
        group_loss_array = [[] for i in range(config.group_size)] # will track the group losses


    total_loss_array = [] # this is for tracking the total loss (it's the sum of group 0 and group 1)
    prob_array = [] # this is for just tracking some probability values
    total_acc_array = [] # this is for tracking the total accuracy


    def loss_function(W, X):

        W_transformed = W.view(config.c, config.d).t()
        output = torch.matmul(X, W_transformed)
        return criterion(output, y).mean()
    

    for epoch in tqdm.tqdm(range(config.num_epochs)):


    
        output = torch.matmul(X, param.view(config.c, config.d).t())
        pred = torch.argmax(output, dim=1)

        prob_array.append(output) # just storing the unnormalized probs for some calculations later

        loss_val_array = criterion(output, y)
        loss = loss_val_array.mean()
        
   
        grad_param = jacfwd(loss_function, argnums=0)(param, X)
    
        H = hessian(loss_function, argnums=0)(param, X)

        for i in range(config.c):
            for j in range(config.c):
                
                if i!=j:
                    H[i*config.d: (i+1)*config.d, j*config.d: (j+1)*config.d] = \
                        torch.zeros(config.d, config.d)


        hessian_inv = torch.inverse(H + damp * torch.eye(H.size(0)))

        param = param - lr * torch.matmul(hessian_inv, grad_param)


        total_loss_array.append(loss_val_array.sum().item()) # the total loss



        with torch.no_grad():

            total_acc_array.append((pred == y).float().mean().item())

            if config.group_size is not None:

                for i in range(config.group_size):
                    group_loss_array[i].append(0)

                for i in range(loss_val_array.shape[0]):

                    group_num = config.group_array[y[i].item()]
                    group_loss_array[group_num][-1] += loss_val_array[i].item()


    return total_loss_array, group_loss_array, total_acc_array, prob_array
 

