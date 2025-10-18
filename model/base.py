from torch import nn


"""
This module introduce a general class that inherit every neding parameter 
for each model.
"""


class PINN(nn.Module):
    def __init__(self,conf):
        super(PINN, self).__init__()
        self.input_dim = conf['input_dim']
        self.hidden_dim = conf['hidden_dim']
        self.output_dim = conf['output_dim']
        self.num_layers = conf['num_layers']
        self.first_omega_0 = conf['first_omega']
        self.hidden_omega_0 = conf['hidden_omega']
        self.model_type = conf['model']
        self.device = conf['device']
        self.sigma = conf['sigma']
        self.mu = conf['mu']
        self.sigma_fourier = conf['sigma_fourier']
