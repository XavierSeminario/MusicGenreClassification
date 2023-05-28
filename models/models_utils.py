import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_curve
import numpy as np
import wandb
import matplotlib.pyplot as plt


def init_weights(model, method = 'Kaiming'):
    """
    Initialization method for the weights and biases of layer of the model.

    Parameters:
    - model: Model that we want to initialize weights and biases.
    - method: Method of initialization. Kaiming (Kaiming Uniform) and Xavier (Xavier Uniform) supported. By default Kaiming is the selected method.
    """

    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            if method == 'Kaiming':
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif method == 'Xavier':
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)      

def calcular_parametres_del_model(model):
    """
    Function that returns the total number of parameters of the model.
    
    Parameters:
    - model: Model whose number of parameters we want to evaluate.

    Return:
    - pytorch_total_params: Number of parameters (int).
    """

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# trainable parameters: {:,}".format(pytorch_total_params))
    return pytorch_total_params



def plot_roc_curve(targets, probas, class_names):
    """
    Plots the roc curve and logs it to wandb.

    Parameters:
    - targets: List containing the ground truth of the genres.
    - probas: List containing the probabilities of predicting each genre for each spectrogram.
    - class_names: List containing the class names.
    """
    y_true = np.array(targets)
    y_probas = np.array(probas)
    fpr = dict()
    tpr = dict()

    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_true,y_probas[...,i], pos_label = i)

        plt.plot(fpr[i], tpr[i], lw=2, label=class_names[i])


    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    wandb.init(project="MusicGenreClassification")

    wandb.log({"chart": plt})


