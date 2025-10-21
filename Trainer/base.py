from .utils import *
import numpy as np
import torch
import os
from utils import generate_grid, save_lora_weights
from torch.utils.tensorboard import SummaryWriter
from utils import AcousticScattering3D
import time
from utils import sample_with_blue_noise
from model.lora import * 
import time
from abc import ABC, abstractmethod
from eval import sound_hard_circle


"""
This module contains bases class for Trainer for 2D and 3D scattering problem.
"""


class LossPINN(ABC):
    """
    Implement the losses used for any training regarding scattering problem solving
    """
    def __init__(self, model, dataloader, loss_fn, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.dataloader = dataloader
        self.device = config['device']

        self.L = config['L']
        self.res = config['res']
        self.Lpml = config['pml_size']
        self.fake_bd = self.L+self.Lpml
        self.a0 = config['pml_dampling']

        self.f = config['frequency']
        self.w = 2*np.pi*self.f
        self.k = self.w / self.config['celerity']  # wavenumber
        self.lora = config.get('lora_train', False)
        self.direction = self.config['direction']

        self.Z_value = self.config.get('Z_value', torch.tensor([0., 0.], device = self.device))
        self.p0  = self.config.get('p0', 1.)
        self.mode = self.config.get('mode', 'plane')
        self.scale = self.config.get('scale', 1.)

        self.weight_bc = 100.
        self.weight_re = 1.
        self.weight_im = 1.

        self.custom_shape = self.config.get('custom_shape', False)
        print(f'Incoming wave mode : {self.mode}') 
        self.bc_loss = self.get_bc_loss()

    def inc_wave(self,x, direction = None):
        """
        Parameters:
            x: [N, d] tensor of observation points
        Returns:
            Tensor of shape [N, d] representing real and imaginary parts.
        """
        if direction == None:
            direction = self.direction
        if self.mode == 'source':
            x_source = self.config['source']
            r = torch.linalg.norm(x - x_source, dim = 1)   #(N,)
            r = r / self.scale
            u = self.p0 * torch.exp(1j * self.k * r)/(4 * np.pi * r)
        else :
            r = x@direction
            u = torch.exp(1j* self.k * r / self.scale)
        return torch.stack((u.real, u.imag), dim=-1)

    def plane_wave_rbc_loss(self, boundary_points, normals, direction = None, forward_method = None):
        """
        Computes Robin boundary condition loss with impedance boundary for a plane wave
        
        Parameters:
        -----------
        boundary_points : torch.Tensor, shape (N, d)
            Coordinates of boundary points where Robin BC is enforced
        normals : torch.Tensor, shape (N, d)
            Outward unit normal vectors at each boundary point
        direction : torch.Tensor, shape (N, d) 
            Direction tensor for hypernetwork training
        forward_method : 
            forward method for hypernetwork training
        Returns:
        --------
        torch.Tensor, scalar
            Combined loss for real and imaginary parts of Robin BC: ∂u/∂n + Z*u = -∂u_inc/∂n - Z*u_inc 
        """
        #print(direction.shape, self.direction.shape)
        if forward_method is not None:
            u, boundary_points = forward_method(boundary_points)
        else : 
            u, boundary_points = self.model(boundary_points, diff = True)

        u_real = u[:, 0]
        u_imag = u[:, 1]
        grads_real = gradient(u_real, boundary_points)
        grads_imag = gradient(u_imag, boundary_points)
        # normal derivatives: dot product with normal vector
        dudn_real = torch.sum(grads_real * normals, dim=1, keepdim=True)
        dudn_imag = torch.sum(grads_imag * normals, dim=1, keepdim=True)
        if direction is None: 
            direction = self.direction
        else:
            direction = direction.unsqueeze(1)

        dot = boundary_points @ direction / self.scale
        duidn = -1j * self.k * torch.exp(1j * self.k * dot) * (normals @ direction)
        if torch.abs(self.Z_value).max() > 0.:
            Zui = -(self.Z_value[0]+1j * self.Z_value[1])  * torch.exp(1j * self.k * dot) 
            rhs_real = torch.real(duidn + Zui).squeeze(1)
            rhs_imag = torch.imag(duidn + Zui).squeeze(1)
            dudn_real = dudn_real.squeeze(1)
            dudn_imag = dudn_imag.squeeze(1)
            loss_real = self.loss_fn(dudn_real + u_real * self.scale * self.Z_value[0] - u_imag  * self.scale * self.Z_value[1],  self.scale * rhs_real)
            loss_imag = self.loss_fn(dudn_imag + u_real * self.scale *self.Z_value[1] + u_imag  * self.scale * self.Z_value[0], self.scale * rhs_imag)
        else:
            rhs_real = torch.real(duidn) 
            rhs_imag = torch.imag(duidn) 
            loss_real = self.loss_fn(dudn_real , self.scale * rhs_real)
            loss_imag = self.loss_fn(dudn_imag, self.scale * rhs_imag)

        return loss_real + loss_imag
    
       
    def source_rbc_loss(self, boundary_points, normals, direction = None, forward_method = None):
        """
        Computes Robin boundary condition loss with impedance boundary for a source wave of intensity
        
        Parameters:
        -----------
        boundary_points : torch.Tensor, shape (N, 2)
            Coordinates of boundary points where Robin BC is enforced
        normals : torch.Tensor, shape (N, 2)
            Outward unit normal vectors at each boundary point
            
        Returns:
        --------
        torch.Tensor, scalar
            Combined loss for real and imaginary parts of Robin BC: ∂u/∂n + Z*u = -∂u_inc/∂n - Z*u_inc 
        """
        u, x = self.model(boundary_points, diff = True)
        u_real = u[:, 0]
        u_imag = u[:, 1]

        grads_real = gradient(u_real, x)
        grads_imag = gradient(u_imag, x)

        x_source = self.config['source']
        r = torch.linalg.norm(x - x_source, dim = 1) / self.scale  #(N,)
        # normal derivatives: dot product with normal vector
        dudn_real = torch.sum(grads_real * normals, dim=1, keepdim=True).squeeze(1)
        dudn_imag = torch.sum(grads_imag * normals, dim=1, keepdim=True).squeeze(1)
        duidx = - (1j * self.k / r**2 - 1 / r**3) * torch.exp(1j * self.k * r) 
        duidx /= (4 * np.pi)
        projection = torch.sum((x - x_source) * normals, dim = 1)   #(N,) 
        if torch.abs(self.Z_value).max() > 0.:
            Zui = (self.Z_value[0]+1j * self.Z_value[1]) * torch.exp(1j * self.k * r)/(4 * np.pi * r)
            rhs_real =  torch.real(duidx) * projection + torch.real(Zui)
            rhs_imag =  torch.imag(duidx) * projection + torch.imag(Zui)
            loss_real = self.loss_fn(dudn_real + u_real * self.scale * self.Z_value[0] - u_imag * self.scale * self.Z_value[1], self.scale * self.p0 * rhs_real)
            loss_imag = self.loss_fn(dudn_imag + u_real * self.scale * self.Z_value[1] + u_imag * self.scale * self.Z_value[0], self.scale * self.p0 * rhs_imag)
        else:
            rhs_real =  torch.real(duidx) * projection 
            rhs_imag =  torch.imag(duidx) * projection 
            loss_real = self.loss_fn(dudn_real, self.scale * rhs_real)
            loss_imag = self.loss_fn(dudn_imag, self.scale * rhs_imag)

        return loss_real + loss_imag
    
        
    def get_bc_loss(self, mode = 'plane'):
        """
        Defines the nature of Robin boundary conditions.
        source -> The incoming wave radiate from a source point.
        plane -> The incoming wave is a plane wave.
        """
        if mode == 'source':
            return self.source_rbc_loss
        elif mode =='plane':
            return self.plane_wave_rbc_loss
        else : 
            raise ValueError(f"Invalid mode '{mode}'. Choose 'source' or 'plane'.")
            
    

class BaseTrainer2D(LossPINN):
    def __init__(self, model, dataloader, loss_fn, config):
        """
        Initialize 2D trainer with model and training configuration
        
        Parameters:
        -----------
        model : torch.nn.Module
            Neural network model (PINN for acoustic scattering)
        dataloader : torch.utils.data.DataLoader
            Dataloader providing training batches
        loss_fn : callable
            Loss function for training optimization
        config : dict
            Configuration dictionary containing training hyperparameters:
            - 'model': str, model name for logging and checkpointing
            - 'preload': bool, whether to load pre-trained weights
            - other training-specific parameters
        """
        super().__init__(model, dataloader, loss_fn, config)
        self.x_grid = generate_grid(self.L, self.res, 2, device=self.device)
        self.init_solution()

    def init_solution(self): 
        '''
        Initialize the solution for a circular shape defined by (center, R)
        '''
        self.tar = torch.zeros((self.res**2,2), device = self.device, dtype = torch.double)
        if not self.custom_shape:
            mask = (self.x_grid[:,0]-self.config['center'][0])**2 + (self.x_grid[:,1]-self.config['center'][1])**2 > self.config['R']**2
            self.tar[mask,:] = sound_hard_circle(self.config, self.x_grid[mask,:] - self.config['center'], self.config['R'])
    
    @abstractmethod
    def train(self):
        pass


    def pml_2d(self, u, x):
        """
        Computes 2D Perfectly Matched Layer (PML) constraint for absorbing boundary conditions
        
        Parameters:
        -----------
        u : torch.Tensor, shape (N, 2)
            Complex wave field as [real, imaginary] components at N points
        x : torch.Tensor, shape (N, 2) 
            Coordinates [x, y] of N points in the computational domain
            
        Returns:
        --------
        torch.Tensor, shape (N, 2)
            PML constraint residual as [real, imaginary] components
            Should be zero for waves satisfying PML wave equation
        """
        # Single jacobian call for all derivatives
        du, _ = jacobian(u, x)

        dudx = du[..., 0]
        dudy = du[..., 1]

        squared_slowness = torch.ones_like(x)    #Can be changed to account modification of sound speed within the computational space
        squared_slowness[..., 1] = 0.
        # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
        dist_west = -torch.clamp(x[..., 0] + (self.fake_bd - self.Lpml), max=0)
        dist_east = torch.clamp(x[..., 0] - (self.fake_bd - self.Lpml), min=0)
        dist_south = -torch.clamp(x[..., 1] + (self.fake_bd - self.Lpml), max=0)
        dist_north = torch.clamp(x[..., 1] - (self.fake_bd - self.Lpml), min=0)

        sx = self.w * self.a0 * ((dist_west / self.Lpml) ** 2 + (dist_east / self.Lpml) ** 2)[..., None]
        sy = self.w * self.a0 * ((dist_north / self.Lpml) ** 2 + (dist_south / self.Lpml) ** 2)[..., None]

        ex = torch.cat((torch.ones_like(sx), sx / self.w), dim=-1)
        ey = torch.cat((torch.ones_like(sy), sy / self.w), dim=-1)

        Ax = compl_div(ey, ex).repeat(1, dudx.shape[-1] // 2)
        Ay = compl_div(ex, ey).repeat(1, dudx.shape[-1] // 2)
        S = compl_mul(ex, ey).repeat(1, dudx.shape[-1] // 2)

        ax, _ = jacobian(compl_mul(Ax, dudx), x)
        ay, _ = jacobian(compl_mul(Ay, dudy), x)

        ax = ax[..., 0]
        ay = ay[..., 1]

        s = compl_mul(compl_mul(S, squared_slowness), self.k** 2 * u)

        pml_constraint = ax + ay + s

        return pml_constraint


class BaseTrainer3D(LossPINN):
    def __init__(self, model, dataloader, loss_fn, config):
        """
        Initialize 3D trainer with model and training configuration
        
        Parameters:
        -----------
        model : torch.nn.Module
            Neural network model (PINN for acoustic scattering)
        dataloader : torch.utils.data.DataLoader
            Dataloader providing training batches
        loss_fn : callable
            Loss function for training optimization
        config : dict
            Configuration dictionary containing training hyperparameters:
            - 'model': str, model name for logging and checkpointing
            - 'preload': bool, whether to load pre-trained weights
            - other training-specific parameters
        """
        super().__init__(model, dataloader, loss_fn, config)
        self.init_solution()

    def init_solution(self):
        if not self.custom_shape: 
            scatterer = AcousticScattering3D(
            sphere_radius = self.dataloader['R'], 
            incident_direction = self.direction.cpu().numpy(),
            device = self.device
            )
            self.solution, self.x_grid = scatterer.total_field_pytorch(self.L, self.res, self.k * self.dataloader['R'])
            self.x_grid = self.x_grid.to(self.device)
            self.solution = self.solution.to(self.device)
            self.solution = torch.stack((self.solution.real, self.solution.imag) , dim = 1)
        elif self.config['hrtf']:
            self.x_grid = generate_grid(self.L, self.res, 3, device=self.device)
            self.solution = torch.zeros_like(self.x_grid)
        else: 
            self.x_grid = generate_grid(self.config['L'], self.config['res'], 3, device = self.device)
            self.eval_grid = np.load('bem_results/evaluation_points.npy')
            print('Evaluation grid shape:', self.x_grid.shape)
            DTYPE = torch.float
            self.eval_grid = torch.tensor(self.eval_grid, device = self.device, dtype = DTYPE)
            self.solution = np.load('bem_results/scattered_field_realimag.npy')
            self.solution = torch.tensor(self.solution, device = self.device, dtype = DTYPE)
        self.solution = self.solution.cpu().numpy()
    
    @abstractmethod
    def train(self):
        pass

    def pml_3D(self, u, x, k = None):
        if k == None:
            k = self.k
        du, _ = jacobian(u, x)

        dudx = du[..., 0]
        dudy = du[..., 1]
        dudz = du[..., 2]

        squared_slowness = torch.ones_like(u)
        squared_slowness[..., 1] = 0.
        # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
        dist_xl = -torch.clamp(x[..., 0] + (self.fake_bd - self.Lpml), max=0)
        dist_xr = torch.clamp(x[..., 0] - (self.fake_bd - self.Lpml), min=0)
        dist_yl = -torch.clamp(x[..., 1] + (self.fake_bd - self.Lpml), max=0)
        dist_yr = torch.clamp(x[..., 1] - (self.fake_bd - self.Lpml), min=0)
        dist_zl = -torch.clamp(x[..., 2] + (self.fake_bd - self.Lpml), max=0)
        dist_zr = torch.clamp(x[..., 2] - (self.fake_bd - self.Lpml), min=0)

        sx = self.w * self.a0 * ((dist_xl / self.Lpml) ** 2 + (dist_xr / self.Lpml) ** 2)[..., None]
        sy = self.w * self.a0 * ((dist_yl / self.Lpml) ** 2 + (dist_yr / self.Lpml) ** 2)[..., None]
        sz = self.w * self.a0 * ((dist_zl / self.Lpml) ** 2 + (dist_zr / self.Lpml) ** 2)[..., None]

        ex = torch.cat((torch.ones_like(sx), sx / self.w), dim=-1)
        ey = torch.cat((torch.ones_like(sy), sy / self.w), dim=-1)
        ez = torch.cat((torch.ones_like(sz), sz / self.w), dim=-1)

        Ax = compl_div(compl_mul(ey, ez), ex).repeat(1, dudx.shape[-1] // 2)
        Ay = compl_div(compl_mul(ez, ex), ey).repeat(1, dudx.shape[-1] // 2)
        Az = compl_div(compl_mul(ex, ey), ez).repeat(1, dudx.shape[-1] // 2)

        S = compl_mul(compl_mul(ex, ey), ez).repeat(1, dudx.shape[-1] // 2)

        ax, _ = jacobian(compl_mul(Ax, dudx), x)
        ay, _ = jacobian(compl_mul(Ay, dudy), x)
        az, _ = jacobian(compl_mul(Az, dudz), x)
        ax = ax[..., 0]
        ay = ay[..., 1]
        az = az[..., 2]
        s = compl_mul(compl_mul(S, squared_slowness), self.scale ** 2 * k** 2 * u)

        pml_constraint = ax + ay + az + s 

        return pml_constraint
