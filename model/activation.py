import torch
from torch import nn
import numpy as np


'''
This module introduces the activation function used for our scattering PINN
'''


class QuadraticTanh(torch.nn.Module):
    def __init__(self, alpha = 1.):
        super().__init__()
        self.alpha = alpha#torch.nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return torch.tanh(x + self.alpha * x **2)

class SineActivation(nn.Module):
    def __init__(self, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, input):
        return torch.sin(self.omega_0 * input)
    

######################################################################################################################################################
#Activation below are not used in the paper 'PHYSICS-INFORMED LEARNING OF NEURAL SCATTERING FIELDS TOWARDS MEASUREMENT-FREE MESH-TO-HRTF ESTIMATION'
######################################################################################################################################################

class GaborActivation(nn.Module):
    def __init__(self, gamma=1.0, omega=30.0):
        super(GaborActivation, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.omega = nn.Parameter(torch.tensor(omega))

    def forward(self, x):
        return torch.exp(-self.gamma * x ** 2) * torch.sin(self.omega * x)

class HyperSine(nn.Module):
    # Comes from "H-SIREN: Improving implicit neural representations with hyperbolic periodic functions". 
    
    def __init__(self, omega_0=30, r = 2):
        super().__init__()
        self.omega_0 = omega_0
        self.r = r
        
    def forward(self, input):
        return torch.sin(self.omega_0 * torch.sinh(self.r * self.linear(input)))
    

class Rowdy(nn.Module):
    def __init__(self, omega_0=30, omega = 1500., num_sin = 4):
        super().__init__()
        self.num_sin = num_sin
        self.omega_0 = omega_0
        self.omega = omega
        self.omegas = nn.Parameter(torch.tensor([self.omega*i for i in range(1,self.num_sin+1)]))  # [num_freqs]
        # Learnable coefficients for combining sinusoids
        self.coeffs = nn.Parameter(torch.zeros(self.num_sin)/self.omega ) # shape must match omegas
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        x = self.linear(input)
        out = x.unsqueeze(-1) * self.omegas         # shape: [N, M, 3]
        out = (self.coeffs * torch.sin(out)).sum(dim = -1)
        return out + torch.sin(self.omega_0 * x)
       