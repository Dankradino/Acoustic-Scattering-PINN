from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import generate_grid, save_lora_weights
from visuals import create_obstacle_patch
from eval import sound_hard_circle
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
from PIL import Image
import time
from model import PhiskModule
from .base import BaseTrainer2D
import torch
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
import re



class PHISK_Trainer2D(BaseTrainer2D, PhiskModule):
    def __init__(self, reference_network, hypernetwork_path, dataloader, loss_fn, config, hconfig):
        self.dim = 2
        self.hidden_dims = config.get('hidden_dims',[128, 256, 512])  
        self.T = config.get('T', None)
    
        super().__init__(reference_network, dataloader, loss_fn, config)
        # Architecture components
        self.reference_network = reference_network
        self.base_network = reference_network                                   # Artefact from previous implementation --- IGNORE ---
        self.hypernetwork_path = hypernetwork_path
        self.hconfig = hconfig
        self.x_grid = generate_grid(self.L, self.res, self.dim, device=self.device)

        if self.hypernetwork_path is not None:
            self.load_hypernetwork_checkpoint(self.hypernetwork_path)

    def train_continuous_direction_control(self):
        """Train the corrective hypernetwork for smooth direction control"""
        print("Training continuous direction control...")
        
        if not hasattr(self, 'direction_interpolation') or self.direction_interpolation is None:
            raise ValueError("LoRA ensemble not loaded. Call load_lora_ensemble() first.")
        weight_reg = 0.01 
        config = self.hconfig['adam']
        num_epochs = config['epochs']
        lr = config['lr']
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']

        # Create optimizer for interpolation and hypernetwork
        params_to_optimize = (list(self.direction_interpolation.parameters()) + 
                            list(self.continuous_hypernetwork.parameters()))
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        
        dataloader_iter = iter(self.dataloader['adam']) 

        direction_keeping = config['direction_keeping_time']
        batch_size = self.hconfig['adam']['batch_size_dir']

        random_directions = self._sample_random_directions(batch_size, ratio = 0.05)
        for epoch in range(num_epochs):
            if epoch% config['keeping_time'] == 0:
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]
            # Sample random directions for training

            if epoch % direction_keeping == 0 :
                random_directions = self._sample_random_directions(batch_size, ratio = min(1, 8 * epoch/num_epochs+0.05))
            
            total_loss = 0
            
            for direction in random_directions:
                # Get predicted parameters
                pred_params, _ = self.get_continuous_direction_params(direction)
                
                # Compute loss using physics-informed loss
                u, x = self.forward_with_lora_params(x_sample, pred_params, diff=True)
                pml_constraint = self.pml_2d(u, x)
                loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
                loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
        
                forward_method = self.create_forward_method(pred_params)
                loss_obstacle = self.bc_loss(boundary_points, normals_points, direction, forward_method)
                loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle

                continuous_deltas = self.continuous_hypernetwork(direction)
                hypernetwork_reg = weight_reg * torch.norm(continuous_deltas)
                
                total_loss += loss + hypernetwork_reg
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            self.writer.add_scalar('Loss/real', loss_real.item(), epoch)
            self.writer.add_scalar('Loss/imag', loss_imag.item(), epoch)
            self.writer.add_scalar('Loss/regularization', hypernetwork_reg.item(), epoch)
            self.writer.add_scalar('Loss/total', total_loss.item(), epoch)
            self.writer.add_scalar('Loss/obstacle', loss_obstacle.item(), epoch)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
                test_direction = torch.tensor([1.0, 0.0], device=self.device)
                _, test_weights = self.get_continuous_direction_params(test_direction)
                print(f"  Test direction [1,0] interpolation weights: {test_weights.detach().cpu().numpy()}")

            if epoch%2000 == 0:
                with torch.no_grad():
                    direction = random_directions[0]
                    pred_params, interpolation_weights = self.get_continuous_direction_params(direction)
                    direction = direction.unsqueeze(1)
                    u = self.forward_with_lora_params(self.x_grid, pred_params).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, direction).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, direction, epoch)

            if epoch%1000 == 0 :
                name = self.config['model']
                save_dir = self.save_dir
                filename = f'{save_dir}{name}_pre'
                self.save_current(filename)

    def fine_tune_direction(self):
        """Train the corrective hypernetwork for smooth direction control"""
        print("Fine tuning continuous direction control...")
        
        weight_reg = 0.01
        if not hasattr(self, 'direction_interpolation') or self.direction_interpolation is None:
            raise ValueError("LoRA ensemble not loaded. Call load_lora_ensemble() first.")
        
        def criterion(x_sample, boundary_points, normals_points, random_directions):
            total_loss = 0
            
            for direction in random_directions:
                # Get predicted parameters
                pred_params, _ = self.get_continuous_direction_params(direction)
                
                # Compute loss using physics-informed loss
                u, x = self.forward_with_lora_params(x_sample, pred_params, diff=True)
                pml_constraint = self.pml_2d(u, x)
                loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
                loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
                forward_method = self.create_forward_method(pred_params)
                loss_obstacle = self.bc_loss(boundary_points, normals_points, direction, forward_method)
                loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
                
                continuous_deltas = self.continuous_hypernetwork(direction)
                hypernetwork_reg = weight_reg * torch.norm(continuous_deltas)
                
                total_loss += loss +  hypernetwork_reg
            
            return total_loss

        def make_closure(x_sample, boundary_points, normals_points, random_directions):
            def closure():
                optimizer.zero_grad()
                loss = criterion(x_sample, boundary_points, normals_points, random_directions)
                loss.backward()
                return loss
            return closure
        
        config = self.hconfig['fine']
        num_epochs = config['epochs']
        lr = config['lr']
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        params_to_optimize = list(self.continuous_hypernetwork.parameters())
        optimizer = torch.optim.LBFGS(params_to_optimize, lr=lr, max_iter = config['max_iter'], line_search_fn = 'strong_wolfe')   
        
        dataloader_iter = iter(self.dataloader['fine']) 

        direction_keeping = config['direction_keeping_time']
        batch_size = self.hconfig['fine']['batch_size_dir']

        random_directions = self._sample_random_directions(batch_size)
            
        for epoch in range(num_epochs):
            if epoch % config['keeping_time'] == 0:
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]
                closure = make_closure(x_sample, boundary_points, normals_points, random_directions)

            if epoch % direction_keeping == 0 :
                random_directions = self._sample_random_directions(batch_size)
                closure = make_closure(x_sample, boundary_points, normals_points, random_directions)
            total_loss = optimizer.step(closure)
            self.writer.add_scalar('Loss/total', total_loss.item(), self.config['adam']['epochs'])

            if epoch% 50 == 0:
                with torch.no_grad():
                    direction = random_directions[0]
                    pred_params, interpolation_weights = self.get_continuous_direction_params(direction)
                    direction = direction.unsqueeze(1)
                    u = self.forward_with_lora_params(self.x_grid, pred_params).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, direction).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, direction, epoch+self.hconfig['adam']['epochs'])

            if epoch%100 == 0 :
                name = self.config['model']
                save_dir = self.save_dir
                filename = f'{save_dir}{name}'
                self.save_current(filename)

    def _sample_random_directions(self, batch_size, ratio = 1.):
        """Sample random directions on unit circle by creating small deviation from evenly spaced points"""
        angles = torch.linspace(0, 2*np.pi, batch_size, device = self.device) + ratio * torch.randn(batch_size, device = self.device)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return directions

    def train(self, save_dir = 'checkpoints/2D/phisk/'):
        """Main training method - placeholder for your training logic"""
        print("Starting enhanced training with LoRA ensemble...")
        start = time.time()
        name = self.config['model']
        log_dir = f'./runs/hyper'
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir = log_dir)
        # Train the continuous direction control system
        self.train_continuous_direction_control()
        # In your save code:
        filename = f'{save_dir}{name}.pth'
        self.save_current(filename)

        print(f'Trained PHISK model saved under  {filename}')

        #Uncomment the following if you want to do L BFGS Training, warning it requires heavy computational ressources
        # self.fine_tune_direction()          
        
        # # In your save code:
        # self.save_current(filename)
        # print(f'Fully trained model saved under {filename}')

        end =  time.time()
        training_time = end - start

        print('Full training time : ', training_time)
        self.writer.close()
        print("Training completed!")

    def save_plot(self, u, u_inc, direction, epoch):
        # Create the figure and subplots
        u_real = u[:, 0].reshape(self.res, self.res)
        u_imag = u[:, 1].reshape(self.res, self.res)
        polygon = self.dataloader['polygon'][:, ::-1]

        self.config['direction'] = direction
        R = self.config['R']
        mask = self.x_grid[:,0]**2 + self.x_grid[:,1]**2 > R**2

        tar = torch.zeros((self.res**2,2), device = self.device, dtype = torch.double)
        tar[mask,:] = sound_hard_circle(self.config, self.x_grid[mask,:], R)
        target = np.zeros((self.res,self.res,2))
        target[...,0] = tar[:,0].cpu().numpy().reshape(self.res,self.res)
        target[...,1] = tar[:,1].cpu().numpy().reshape(self.res,self.res)


        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        # Real part plot
        im0 = axs[0, 0].imshow(u_real, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Scattered field real part')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0,1].imshow(u_imag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[0,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0,1].set_title('Scattered field imag part')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part plot
        im2 = axs[1, 0].imshow(target[...,0]-u_real , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('Difference with groundtruth real part')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part plot
        im3 = axs[1,1].imshow(target[...,1]-u_imag , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1,1].set_title('Difference with groundtruth imag part')
        fig.colorbar(im3, ax=axs[1, 1])

        plt.tight_layout()

        # Save the figure to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Open the image from the buffer
        image = Image.open(buf)
        image.load() 
        image = np.asarray(image)
        image = image.copy()
        # Convert to tensor (C x H x W format)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0 # Add a batch dimension
        # Log the image to TensorBoard
        self.writer.add_image(f'Scattered_field_pred/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard

        # Close the plot and the buffer
        plt.close()
        buf.close()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        u_sc = np.sqrt(u_real**2+u_imag**2)
        u_full = np.linalg.norm(u + u_inc, axis = 1).reshape(self.res,self.res)

        # Real part plot
        im0 = axs[0].imshow(u_sc, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0].set_title('Scattered field')
        fig.colorbar(im0, ax=axs[0])

        # Imaginary part plot
        im1 = axs[1].imshow(u_full, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1].set_title('Full field')
        fig.colorbar(im1, ax=axs[1])

        plt.tight_layout()

        # Save the figure to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Open the image from the buffer
        image = Image.open(buf)
        image.load() 
        image = np.asarray(image)
        image = image.copy()
        # Convert to tensor (C x H x W format)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0 # Add a batch dimension

        # Log the image to TensorBoard
        self.writer.add_image(f'Full_field_pred/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard

        # Close the plot and the buffer
        plt.close()
        buf.close()

        # Create the figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))

        # Close the plot and the buffer
        plt.close()
        buf.close()




# Initialization function
def initialize_phisk_trainer2D(reference_network, hypernetwork_path, dataloader, loss_fn, config, hconfig, lora_dir):
    """
    Initialize the PHISK trainer with LoRA ensemble
    """
    trainer = PHISK_Trainer2D(reference_network, hypernetwork_path, dataloader, loss_fn, config, hconfig)
    
    # Load LoRA ensemble
    if not hconfig['load']:
        trainer.load_lora_ensemble(lora_dir)
    
    return trainer

