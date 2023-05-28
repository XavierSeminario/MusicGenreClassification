import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm
from models.models import *
from models.models_utils import *

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
loss = nn.CrossEntropyLoss()


spectrograms_list, genres_list = LoadDataPipeline()

train_dataloader,test_dataloader, targets = CreateTrainTestLoaders(spectrograms_list, genres_list, train_size, 
                                                        train_kwargs, test_kwargs, False)


path_model = "./modelsguardats/CNN"

model = RNN()
model.load_state_dict(torch.load(path_model))
model.eval()

loss_test_epoch, prediction, probas = test(model, device, test_dataloader, loss)

print("Loss del model: ", loss_test_epoch)
print("Prediccions: ", prediction)
print("Probabilitats: ", probas)