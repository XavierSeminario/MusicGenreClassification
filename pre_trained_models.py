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

path_model = "./modelsguardats/CNN"

model = torch.load(path_model, map_location='cpu')
