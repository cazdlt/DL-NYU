#Tomada de https://github.com/Atcold/pytorch-Deep-Learning

import random
import torch
from torch import nn
import math
from lib.plot_lib import plot_data, plot_model, set_default
import matplotlib.pyplot as plt

#Inicialización
set_default()
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(N,D,C):
    # Data de entrada
    X = torch.zeros(N * C, D).to(device)
    y = torch.zeros(N * C, dtype=torch.long).to(device)
    for c in range(C):
        index = 0
        t = torch.linspace(0, 1, N)
        # When c = 0 and t = 0: start of linspace
        # When c = 0 and t = 1: end of linpace
        # This inner_var is for the formula inside sin() and cos() like sin(inner_var) and cos(inner_Var)
        inner_var = torch.linspace(
            # When t = 0
            (2 * math.pi / C) * (c),
            # When t = 1
            (2 * math.pi / C) * (2 + c),
            N
        ) + torch.randn(N) * 0.2
        
        for ix in range(N * c, N * (c + 1)):
            X[ix] = t[index] * torch.FloatTensor((
                math.sin(inner_var[index]), math.cos(inner_var[index])
            ))
            y[ix] = c
            index += 1

    print("Shapes:")
    print("X:", tuple(X.size()))
    print("y:", tuple(y.size()))

    return X,y

def create_model(D,H,C):
    model = nn.Sequential(
        nn.Linear(D, H),
        nn.ReLU(),
        nn.Linear(H, C)
    )
    model.to(device)

    return model

def train_model(model,num_epochs,learning_rate):
    # nn package also has different loss functions.
    # we use cross entropy loss for our classification task
    criterion = torch.nn.CrossEntropyLoss()

    # we use the optim package to apply
    # ADAM for our parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2

    # Training
    for t in range(num_epochs):
        
        # Feed forward to get the logits
        y_pred = model(X)
        
        # Compute the loss and accuracy
        loss = criterion(y_pred, y)
        score, predicted = torch.max(y_pred, 1)
        acc = (y == predicted).sum().float() / len(y)
        print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
        
        # zero the gradients before running
        # the backward pass.
        optimizer.zero_grad()
        
        # Backward pass to compute the gradient
        # of loss w.r.t our learnable params. 
        loss.backward()
        
        # Update params
        optimizer.step()
        

if __name__ == "__main__":
    
    #initialize parameters
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    N = 1000  # num_samples_per_class
    D = 2  # number of features
    C = 3  # number of output classes
    H = 100  # number of hidden units for the two layer neural net
    num_epochs=1500 #number of training epochs

    #create training data
    X,y=create_dataset(N,D,C)
    plot_data(X,y)
    plt.show()

    #define model
    model=create_model(D,H,C)
    print(model)

    #train model
    train_model(model,num_epochs,learning_rate)
    plot_model(X, y, model)
    plt.show()

