from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import generate_grid
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
from PIL import Image
import time

import torch
import numpy as np
import glob
from collections import OrderedDict
import re
from model import PhiskModule
from .base import BaseTrainer3D


class PHISK_Trainer3D(BaseTrainer3D, PhiskModule):
    def __init__(self, base_network, hypernetwork_path, dataloader, loss_fn, config, hconfig):
        # Architecture components
        self.dim = 3
        self.hidden_dims = config.get('hidden_dims',[128, 256, 512])   
        self.T = config.get('T', None)
        self.num_fourier_features = config.get('num_fourier_features', 64)
        super().__init__(base_network, dataloader, loss_fn, config)
        self.base_network = base_network
        self.hypernetwork_path = hypernetwork_path
        self.hconfig = hconfig
        self.x_grid = generate_grid(self.L, self.res, self.dim, device=self.device)
        
        # New components for continuous direction control
        if self.hypernetwork_path is not None:
            self.load_hypernetwork_checkpoint(self.hypernetwork_path)


    def train_continuous_direction_control(self):
        """Train the interpolation mechanism and hypernetwork for smooth direction control"""
        print("Training continuous direction control...")
        
        if not hasattr(self, 'direction_interpolation') or self.direction_interpolation is None:
            raise ValueError("LoRA ensemble not loaded. Call load_lora_ensemble() first.")

        weight_reg = 0.01  #Originally 0.001
        
        config = self.hconfig['adam']
        num_epochs = config['epochs']
        lr = config['lr']
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']

        # Create optimizer for interpolation and hypernetwork
        params_to_optimize = (list(self.direction_interpolation.parameters()) +    #If you want to change the interpolation module
                            list(self.continuous_hypernetwork.parameters()))
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        
        dataloader_iter = iter(self.dataloader['adam']) 

        direction_keeping = config['direction_keeping_time']

        batch_size = self.hconfig['adam']['batch_size_dir']

        random_directions = self._sample_random_directions(batch_size, ratio = 0.2)

        for epoch in range(num_epochs):
            if epoch% config['keeping_time'] == 0:
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]
            # Sample random directions for training

            if epoch % direction_keeping == 0 :
                random_directions = self._sample_random_directions(batch_size, ratio = min(1, 8 * epoch/num_epochs + 0.2))
            
            total_loss = 0
            
            for direction in random_directions:
                # Get predicted parameters
                pred_params, interpolation_weights = self.get_continuous_direction_params(direction)
                
                # Compute loss using physics-informed loss
                u, x = self.forward_with_lora_params(x_sample, pred_params, diff=True)
                pml_constraint = self.pml_3d(u, x)
                loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
                loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
                
                loss_obstacle = self.neumann_bc_loss(boundary_points, normals_points, direction, pred_params)
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
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

            if epoch%500 == 0:
                with torch.no_grad():
                    direction = random_directions[0]
                    pred_params, _ = self.get_continuous_direction_params(direction)
                    direction = direction.unsqueeze(1)
                    u = self.forward_with_lora_params(self.x_grid, pred_params).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, direction).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch)

            if epoch%1000 == 0 :
                name = self.config['model']
                save_dir = self.save_dir
                filename = f'{save_dir}{name}_pre.pth'
                self.save_current(filename)



    def fine_tune_direction(self):
        """Train the interpolation mechanism and hypernetwork for smooth direction control"""
        print("Fine tuning continuous direction control...")
        
        if not hasattr(self, 'direction_interpolation') or self.direction_interpolation is None:
            raise ValueError("LoRA ensemble not loaded. Call load_lora_ensemble() first.")
        weight_reg = 0.01
        def criterion(x_sample, boundary_points, normals_points, random_directions):
            total_loss = 0
            
            for direction in random_directions:
                # Get predicted parameters
                pred_params, interpolation_weights = self.get_continuous_direction_params(direction)
                
                # Compute loss using physics-informed loss
                u, x = self.forward_with_lora_params(x_sample, pred_params, diff=True)
                pml_constraint = self.pml_3d(u, x)
                loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
                loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
                
                loss_obstacle = self.neumann_bc_loss(boundary_points, normals_points, direction, pred_params)
                loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
                
                continuous_deltas = self.continuous_hypernetwork(direction)
                hypernetwork_reg = weight_reg * torch.norm(continuous_deltas)
                
                total_loss += loss + hypernetwork_reg
            
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

        optimizer = torch.optim.LBFGS(self.continuous_hypernetwork.parameters(), lr=lr, max_iter = config['max_iter'], line_search_fn = 'strong_wolfe')   
        
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

            # Backward pass
            
            total_loss = optimizer.step(closure)

            self.writer.add_scalar('Loss/total', total_loss.item(), self.config['adam']['epochs'])

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

            if epoch%50 == 0:
                with torch.no_grad():
                    direction = random_directions[0]
                    pred_params, interpolation_weights = self.get_continuous_direction_params(direction)
                    direction = direction.unsqueeze(1)
                    u = self.forward_with_lora_params(self.x_grid, pred_params).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, direction).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch+self.config['adam']['epochs'])


            if epoch%100 == 0 :
                name = self.config['model']
                filename = f'checkpoints/hyper3D/{name}_{epoch}.pth'
                self.save_current(filename)



    def _sample_random_directions(self, batch_size, ratio=1.0):
        """
        Sample evenly spread directions on the 3D unit sphere,
        with optional Gaussian noise (scaled by `ratio`).
        """
        # Golden angle in radians
        golden_angle = np.pi * (3 - np.sqrt(5))

        # Indices
        i = torch.arange(batch_size, dtype=torch.float32, device=self.device)

        # z goes from 1 to -1
        z = 1 - (2 * i + 1) / batch_size
        radius = torch.sqrt(1 - z ** 2)

        theta = golden_angle * i

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        directions = torch.stack([x, y, z], dim=1)

        if ratio > 0:
            noise = ratio * torch.randn_like(directions)
            directions = directions + noise
            directions = torch.nn.functional.normalize(directions, p=2, dim=1)

        return directions
    

    def visualize_direction_interpolation(self, direction1, direction2, num_steps=10):
        """Visualize smooth interpolation between two directions"""
        print(f"Visualizing interpolation from {direction1.cpu().numpy()} to {direction2.cpu().numpy()}")
        
        interpolation_results = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            # Linear interpolation in direction space
            interp_direction = (1 - alpha) * direction1 + alpha * direction2
            # Normalize to unit vector
            interp_direction = interp_direction / torch.norm(interp_direction)
            
            # Get parameters for interpolated direction
            pred_params, interpolation_weights = self.get_continuous_direction_params(interp_direction)
            
            interpolation_results.append({
                'direction': interp_direction.cpu().numpy(),
                'interpolation_weights': interpolation_weights.detach().cpu().numpy(),
                'alpha': alpha
            })
            
            print(f"  Step {i}: direction={interp_direction.cpu().numpy()}, "
                  f"max_interpolation={interpolation_weights.max().item():.3f}")
        
        return interpolation_results
    
    def train(self, save_dir = 'checkpoints/3D/phisk/'):
        """Main training method - placeholder for your training logic"""
        print("Starting enhanced training with LoRA ensemble...")
        start = time.time()
        name = self.config['model']
        log_dir = f'./runs/hyper'
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir = log_dir)
        # Train the continuous direction control system

        self.train_continuous_direction_control()
        filename = f'{save_dir}{name}.pth'
        self.save_current(filename)

        print(f'Pretrained model saved under  {filename}')

        #Uncomment the following if you want to do L BFGS Training, warning it requires heavy computational ressources (experiment show it does not really improve results if you don't use really huge batch size)
        #self.fine_tune_direction()
        # # In your save code:
        # self.save_current(filename)
        # print(f'Fully trained model saved under {filename}')

        end =  time.time()
        training_time = end - start

        print('Full training time : ', training_time)
        self.writer.close()
        print("Training completed!")


    def compute_laplacian_fd(self, u):
        # Create an output array initialized to zero
        lap = np.zeros_like(u)

        # Compute central differences (skip boundaries)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[0:-2, 1:-1]) * self.res**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, 0:-2]) * self.res**2
        )

        return lap

    def save_plot(self, u, u_inc, epoch):
        # Create the figure and subplots
        z_choice = self.res//2
        format = (slice(None), slice(None), z_choice)
        u_real = u[:, 0].reshape(self.res, self.res, self.res)
        u_imag = u[:, 1].reshape(self.res, self.res, self.res)
        u_real = u_real[format]
        u_imag = u_imag[format]
        u_sc = np.sqrt(u_real**2+u_imag**2)


        u_full = np.linalg.norm(u + u_inc, axis = 1).reshape(self.res, self.res, self.res)
        u_full = u_full[format]

        u_rez = u_real.copy()
        u_imz = u_imag.copy()

        u_rey = u[:, 0].reshape(self.res, self.res, self.res)
        u_imy = u[:, 1].reshape(self.res, self.res, self.res)
        u_rey = u_rey[: , z_choice ,:]
        u_imy = u_imy[: , z_choice ,:]

        u_rex = u[:, 0].reshape(self.res, self.res, self.res)
        u_imx = u[:, 1].reshape(self.res, self.res, self.res)
        u_rex = u_rex[z_choice , :, :]
        u_imx = u_imx[z_choice , :, :]

        # Apply masks to hide mesh regions
        xy_mask = self.dataloader['mesh_mask'][:, :, z_choice]
        xz_mask = self.dataloader['mesh_mask'][:, z_choice, :]  
        yz_mask = self.dataloader['mesh_mask'][z_choice, :, :]

        u_rez = np.ma.masked_where(xy_mask, u_rez)
        u_imz = np.ma.masked_where(xy_mask, u_imz)
        u_rex = np.ma.masked_where(yz_mask, u_rex)
        u_imx = np.ma.masked_where(yz_mask, u_imx)
        u_rey = np.ma.masked_where(xz_mask, u_rey)
        u_imy = np.ma.masked_where(xz_mask, u_imy)

        u_sc = np.ma.masked_where(xy_mask, u_sc)
        u_full = np.ma.masked_where(xy_mask, u_full)

        fig, axs = plt.subplots(3, 2, figsize=(12, 5))
        # Real part plot
        im0 = axs[0, 0].imshow(u_rez, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('XY Real part')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0,1].imshow(u_imz, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0,1].set_title('XY Imag part')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part plot
        im2 = axs[1, 0].imshow(u_rey , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1, 0].set_title('XZ Real part')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part plot
        im3 = axs[1,1].imshow(u_imy , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1,1].set_title('XZ Imag part')
        fig.colorbar(im3, ax=axs[1, 1])
        plt.tight_layout()

        # Real part plot
        im4 = axs[2, 0].imshow(u_rex , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[2, 0].set_title('YZ Real part')
        fig.colorbar(im4, ax=axs[2, 0])

        # Imaginary part plot
        im5 = axs[2,1].imshow(u_imx , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[2,1].set_title('YZ Imag part')
        fig.colorbar(im5, ax=axs[2, 1])
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
        self.writer.add_image(f'Scattered_field_HRTF/{epoch}', image_tensor[0], 0)


        # Close the plot and the buffer
        plt.close()
        buf.close()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Real part plot
        im0 = axs[0].imshow(np.log(u_sc), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[0].set_title('Scattered field')
        fig.colorbar(im0, ax=axs[0])

        # Imaginary part plot
        im1 = axs[1].imshow(np.log(u_full), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
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
        self.writer.add_image(f'Full_field_HRTF/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard

        # Close the plot and the buffer
        plt.close()
        buf.close()

        # Create the figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        lap_u_real = self.compute_laplacian_fd(u_real)
        lap_u_imag = self.compute_laplacian_fd(u_imag)

        k = self.k #.item() 
        res_pde_real = np.abs(lap_u_real - k ** 2 * self.config['scale'] ** 2 * u_real) + 1e-6 
        res_pde_imag = np.abs(lap_u_imag - k ** 2 * self.config['scale'] ** 2 * u_imag) + 1e-6

        res_pde_real = np.log(res_pde_real)
        res_pde_imag = np.log(res_pde_imag)

        res_pde_real = np.ma.masked_where(xy_mask, res_pde_real)
        res_pde_imag = np.ma.masked_where(xy_mask, res_pde_imag)
        # Real part plot
        im0 = axs[0].imshow(res_pde_real, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0].set_title('Residual of real part')
        fig.colorbar(im0, ax=axs[0])

        # Imaginary part plot
        im1 = axs[1].imshow(res_pde_imag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1].set_title('Residual of imaginary part')
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
        self.writer.add_image(f'Residuals_HRTF/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard

        # Close the plot and the buffer
        plt.close()
        buf.close()





# Initialization function
def initialize_phisk_trainer3D(base_network, hypernetwork_path, dataloader, loss_fn, config, hconfig, lora_dir):
    """
    Initialize the PHISK trainer with LoRA ensemble
    """
    trainer = PHISK_Trainer3D(base_network, hypernetwork_path, dataloader, loss_fn, config, hconfig)
    
    # Load LoRA ensemble
    if not hconfig['load']:
        trainer.load_lora_ensemble(lora_dir)
    
    return trainer
    





