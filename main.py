import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm
from models.models import *

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 30
batch_size = 50        # number of samples during training
test_batch_size = 50  # number of samples for test 
train_size = 0.8

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}



"""
def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model
"""
if __name__ == "__main__":
    os.system('./download_data.sh')
    wandb.login()
    with wandb.init(project="MusicGenreClassification"):

        spectrograms_list, genres_list = LoadDataPipeline()

        train_dataloader,test_dataloader = CreateTrainTestLoaders(spectrograms_list, genres_list, train_size, 
                                                                train_kwargs, test_kwargs)
        
        print(test_dataloader)
        print("Creacion Modelo")
        model = CNNGH()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #Scheduler that will modify the learning ratio dinamically according to the test loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        model.to(device)
        print("Inicio epochs")

        for epoch in range(1, epochs + 1):
            loss_train_epoch = train(model, device, train_dataloader, optimizer, loss, epoch)
            loss_test_epoch, prediction = test(model, device, test_dataloader, loss)
        #config = dict(
        #   epochs=5,
        #  classes=10,
        # kernels=[16, 32],
        # batch_size=128,
        # learning_rate=5e-3,
        # dataset="MNIST",
        # architecture="CNN")
        #model = model_pipeline(config)