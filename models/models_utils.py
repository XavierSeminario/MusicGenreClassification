import torch.nn as nn
import torch.nn.functional as F
import torch
import torchviz

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


def mostra_estructura_model_torchviz(model):
    """
    Function that allows the visualization of the structure of the model as a diagram.

    Parameters:
    - model: Model whose structure we want to evaluate.

    Return:
    - The model diagram.
    """
    
    from torchviz import make_dot
    test_input = torch.randn(1, 1, 28, 28)
    return make_dot(model(test_input), params=dict(model.named_parameters()))
