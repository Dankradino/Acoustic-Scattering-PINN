import torch
from torch import nn
import numpy as np
from .base import PINN
from .activation import SineActivation

'''
This module implements SIREN related architectures for scattering PINN.
'''


class SineLayer(nn.Module):
    # See 'Implicit Neural Representations with Periodic Activation Functions' (Sitzmann and al 2020) for more details.
    # The implementation here differ from integrating Sine activation as an independant Layer instead of integrating it into SineLayer
    # The weight initialization is crucial here that's why we still denote it as SineLayer    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
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
        return self.linear(input)


class Siren(nn.Module):
    '''
    Siren network with SineLayer as primary component.
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.activation = SineActivation
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.net.append(self.activation(omega_0=first_omega_0))


        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
            self.net.append(self.activation(omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, diff= False):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        if diff:
            return output, coords        
        return output     

class Siren_integrated(nn.Module):
    """
    Integrated Siren module inside other Networks.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.activation = SineActivation

        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        self.net.append(self.activation(omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
            self.net.append(self.activation(omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output




class SirenPINN(PINN):
    ''' PINN using SIREN as main network.'''
    def __init__(self, config):
        super(SirenPINN, self).__init__(config)
        self.net = Siren(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, first_omega_0=self.first_omega_0, hidden_omega_0 = self.hidden_omega_0)
    def forward(self, x, diff = False):
        return self.net(x, diff)
        
class SirenDouble(PINN):
    """
    Siren based on predicting each part real and imag independently"""
    def __init__(self, config):
        super(SirenDouble, self).__init__(config)
        self.output_dim = self.output_dim//2
        self.siren_re = Siren_integrated(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, first_omega_0=self.first_omega_0, hidden_omega_0 = self.hidden_omega_0)
        self.siren_im = Siren_integrated(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, first_omega_0=self.first_omega_0, hidden_omega_0 = self.hidden_omega_0)
    def forward(self, coords, diff = False):
        coords = coords.clone().detach().requires_grad_(True)
        real = self.siren_re(coords)
        im = self.siren_im(coords)
        output =  torch.stack((real, im), dim = 1).squeeze(-1)
        if diff:
            return output, coords
        return output


class SirenMagPhase(PINN):
    """
    Siren based on the magnitude/phase decomposition instead of real/imag part for complex number
    """
    def __init__(self, config):
        super(SirenMagPhase, self).__init__(config)
        self.output_dim = 3
        self.net = Siren_integrated(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, first_omega_0=self.first_omega_0, hidden_omega_0 = self.hidden_omega_0)

    def forward(self, coords, diff = False):
        coords = coords.clone().detach().requires_grad_(True)
        polar_decomposition = self.net(coords)
        
        r = torch.exp(polar_decomposition[..., 0])
        u = polar_decomposition[..., 1]
        v = polar_decomposition[..., 2]
        phi = torch.atan2(u, v)
        output = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim = 1) 
        if diff:
            return output, coords
        return output
    

class SirenConditioned(PINN):
    '''
    SIREN using two input : coordinate and direction, used for PHISK comparison.
    '''
    def __init__(self, config):
        super(SirenConditioned, self).__init__(config)
        self.input_dim = self.input_dim * 2
        self.siren = Siren_integrated(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, first_omega_0=self.first_omega_0, hidden_omega_0 = self.hidden_omega_0)
    
    def forward(self, coords, direction, diff = False):
        coords = coords.clone().detach().requires_grad_(True)
        # Concatenate along last dimension
        coords_augmented = torch.cat([coords, direction.expand(coords.size(0), -1)], dim=1)
        output = self.siren(coords_augmented)
        if diff : 
            return output,coords
        return output
