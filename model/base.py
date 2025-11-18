from torch import nn


"""
This module introduce a general class that inherit every neding parameter 
for each model.
"""


class PINN(nn.Module):
    def __init__(self,config):
        super(PINN, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config['num_layers']
        self.first_omega_0 = config['first_omega']
        self.hidden_omega_0 = config['hidden_omega']
        self.model_type = config['model']
        self.device = config['device']
        self.sigma_fourier = config['sigma_fourier']
