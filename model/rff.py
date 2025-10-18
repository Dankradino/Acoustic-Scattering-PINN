import torch
from torch import nn
import numpy as np
from .base import PINN
from .activation import QuadraticTanh

'''
Implements the Random Fourier Features related PINN
'''


class PINN_RFF(PINN):
    """Physics-Informed Neural Network with Random Weight Factorization layers"""
    
    def __init__(self, config, activation = QuadraticTanh(alpha = 1.)):
        super(PINN_RFF, self).__init__(config)
        self.num_fourier_features = config['rff']
        B = self.sigma_fourier * torch.randn(
            (self.input_dim, self.num_fourier_features), 
            device=self.device
            )
        self.register_buffer("B", B) 

        # First layer: Fourier features to hidden
        first_layer = nn.Linear(2 * self.num_fourier_features, self.hidden_dim)
        
        # Hidden layers with alternating MLP and activation
        other_layers = []
        other_layers.append(activation)
        
        for _ in range(self.num_layers - 1):
            other_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim)) 
            other_layers.append(activation)
        
        # Final layer: hidden to output
        final_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Construct the network
        self.net = nn.Sequential(first_layer, *other_layers, final_layer)   
        
    def forward(self, coords, diff=False):
        """Forward pass through the network
        
        Args:
            coords: Input coordinates
            diff: Whether to return gradients for differentiation
        """
        coords = coords.clone().detach().requires_grad_(True)
        output = self.fourier_feature_mapping(coords)
        output = self.net(output)
        
        if diff:
            return output, coords
        return output
    
    def fourier_feature_mapping(self, x):
        """Apply Fourier feature mapping to input coordinates"""
        return torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi*x @ self.B)], dim=-1)



class Polar_RFF(PINN):
    """Physics-Informed Neural Network with Random Weight Factorization layers"""
    
    def __init__(self, config, activation = QuadraticTanh(alpha = 1.)):
        super(Polar_RFF, self).__init__(config)
        self.num_fourier_features = config['rff'] 
        B = self.sigma_fourier * torch.randn(
            (self.input_dim, self.num_fourier_features), 
            device=self.device
            )
        self.output_dim = 3 #FIXED 
        self.register_buffer("B", B) 
        activation = activation #Shared alpha parameter across all layers to be more physically consistent

        # First layer: Fourier features to hidden
        first_layer = nn.Linear(2 * self.num_fourier_features, self.hidden_dim) #RWF(2 * self.num_fourier_features, self.hidden_dim, self.mu, self.sigma)
        
        # Hidden layers with alternating MLP and activation
        other_layers = []
        other_layers.append(activation)
        
        for _ in range(self.num_layers - 1):
            other_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim)) 
            other_layers.append(activation)
        
        # Final layer: hidden to output
        final_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Construct the network
        self.net = nn.Sequential(first_layer, *other_layers, final_layer)   
        
    def forward(self, coords, diff=False):
        """Forward pass through the network
        
        Args:
            coords: Input coordinates
            diff: Whether to return gradients for differentiation
        """
        coords = coords.clone().detach().requires_grad_(True)
        polar_decomposition = self.fourier_feature_mapping(coords)
        polar_decomposition = self.net(polar_decomposition)
        r = torch.exp(polar_decomposition[..., 0])
        u = polar_decomposition[..., 1]
        v = polar_decomposition[..., 2]
        phi = torch.atan2(u, v)
        output = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim = 1) 
        if diff:
            return output, coords
        return output
    
    def fourier_feature_mapping(self, x):
        """Apply Fourier feature mapping to input coordinates"""
        return torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi*x @ self.B)], dim=-1)


class PINN_directional_RFF(PINN):
    """Physics-Informed Neural Network with Random Weight Factorization layers"""
    
    def __init__(self, config):
        super(PINN_directional_RFF, self).__init__(config)
        self.num_fourier_features =  config['rff'] #512#64
        B = self.sigma_fourier * torch.randn(
            (2 * self.input_dim, self.num_fourier_features), 
            device=self.device
            )
        self.register_buffer("B", B) 
        activation = QuadraticTanh(alpha = 1.)

        # First layer: Fourier features to hidden
        first_layer = nn.Linear(2 * self.num_fourier_features, self.hidden_dim) #RWF(2 * self.num_fourier_features, self.hidden_dim, self.mu, self.sigma)
        
        # Hidden layers with alternating RWF and activation
        other_layers = []
        other_layers.append(activation)
        
        for _ in range(self.num_layers - 1):
            other_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim)) #RWF(self.hidden_dim, self.hidden_dim, self.mu, self.sigma))
            other_layers.append(activation)
        
        # Final layer: hidden to output
        final_layer = nn.Linear(self.hidden_dim, self.output_dim) #RWF(self.hidden_dim, self.output_dim, self.mu, self.sigma)
        
        # Construct the network
        self.net = nn.Sequential(first_layer, *other_layers, final_layer)   #Rename to net
        
        
    def forward(self, coords, direction, diff=False):
        """Forward pass through the network
        
        Args:
            coords: Input coordinates
            diff: Whether to return gradients for differentiation
        """
        coords = coords.clone().detach().requires_grad_(True)
        input = torch.cat([coords, direction.expand(coords.size(0), -1)], dim=1)
        output = self.fourier_feature_mapping(input)
        output = self.net(output)
        
        if diff:
            return output, coords
        return output
    
    def fourier_feature_mapping(self, x):
        """Apply Fourier feature mapping to input coordinates"""
        #print(x.shape, self.B.shape)
        return torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi*x @ self.B)], dim=-1)







######################################################################################################################################################
#Model defined below are not used in the paper 'PHYSICS-INFORMED LEARNING OF NEURAL SCATTERING FIELDS TOWARDS MEASUREMENT-FREE MESH-TO-HRTF ESTIMATION'
######################################################################################################################################################


class RBFPinn(PINN):
    """Physics-Informed Neural Network with Random Weight Factorization layers"""
    
    def __init__(self, config):
        super(RBFPinn, self).__init__(config)

        self.num_features =  config['rff'] 

        self.features_mapping_name = config['map_name']
        self.feature_mapping = RBF(self.num_features, self.features_mapping_name, self.input_dim)
        activation = QuadraticTanh()

        # First layer: Fourier features to hidden
        first_layer = nn.Linear(self.num_features, self.hidden_dim) 
        
        # Hidden layers with alternating RBF and activation
        other_layers = []
        other_layers.append(activation)
        
        for _ in range(self.num_layers - 1):
            other_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim)) 
            other_layers.append(activation)
        
        # Final layer: hidden to output
        final_layer = nn.Linear(self.hidden_dim, self.output_dim) 
        
        # Construct the network
        self.RFF = nn.Sequential(first_layer, *other_layers, final_layer)
        
    def forward(self, coords, diff=False):
        """Forward pass through the network
        
        Args:
            coords: Input coordinates
            diff: Whether to return gradients for differentiation
        """
        coords = coords.clone().detach().requires_grad_(True)
        output = self.feature_mapping(coords)
        output = self.RFF(output)
        
        if diff:
            return output, coords
        return output



class RBF(nn.Module):
    '''
    Implements different features embedding than random fourier features using
    radial basis function.
    '''
    def __init__(self, num_features, features_mapping_name, input_dim):
        super(RBF, self).__init__()
        self.eps = 1e-12
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features_mapping_name = features_mapping_name
        self.num_features = num_features
        self.input_dim = input_dim
        self.log_sigma = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.kappa = 4.0
        if self.input_dim == 2:
            self.mus = torch.linspace(-math.pi, math.pi, self.num_features).to(device)
        else : 
            self.mus = torch.linspace(-math.pi, math.pi, self.num_features // 2).to(device)
        self.kernel = self.select_kernel()

    def select_kernel(self):
        if self.features_mapping_name == 'cubic' :
            return lambda r : r**3
        elif self.features_mapping_name == 'tps' :
            return lambda r : r**2 * torch.log(r + self.eps)
        elif self.features_mapping_name == 'ga' :
            return lambda r: torch.exp(- r**2 / (torch.exp(self.log_sigma)**2))
        elif self.features_mapping_name == 'mq' :
            return lambda r : torch.sqrt(1 + r**2 / (torch.exp(self.log_sigma)**2))
        elif self.features_mapping_name == 'imq' :
            return lambda r : 1 / torch.sqrt(1 + r**2 / (torch.exp(self.log_sigma)**2))
        elif self.features_mapping_name == 'vmbf': 
            return self.vmbf_features()
        else:
            print('No kernel found')
        
    def vmbf_features(self): 
        def vmbf_2D(x):
            return torch.exp(
            self.kappa * torch.cos(
            torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1) - self.mus))
        
        def vmbf_3D(x):
            theta = torch.atan2(x[:, 1], x[:, 0])               # azimuth
            phi = torch.atan2(x[:, 2], torch.norm(x[:, :2], dim=1))  # elevation

            theta_feat = torch.exp(self.kappa * torch.cos(theta.unsqueeze(1) - self.mus))
            phi_feat = torch.exp(self.kappa * torch.cos(phi.unsqueeze(1) - self.mus))

            features = torch.cat([theta_feat, phi_feat], dim=1)  # (B, num_features)
            return features
    
        if self.input_dim == 2:
            return vmbf_2D
        else :
            return vmbf_3D


    def forward(self, x):
        if self.features_mapping_name == 'vmbf':
            phi = self.kernel(x)
            r = torch.norm(x, dim=1, keepdim=True)  # shape (B, 1)
            phi = phi * r  
        else :
            x = x.unsqueeze(1)
            r = torch.linalg.norm(x-self.c, dim = 2)
            phi = self.kernel(r)
        features = phi / (phi.sum(dim = 1, keepdim = True) + self.eps)
        return  features
   