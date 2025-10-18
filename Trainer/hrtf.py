from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from pysofaconventions.SOFAFile import SOFAFile
from torch.optim import LBFGS
from io import BytesIO
from PIL import Image
from utils import generate_grid
import yaml
from utils import sample_with_blue_noise
from .base import BaseTrainer3D

class HRTFTrainer(BaseTrainer3D):
    def __init__(self, model, dataloader, loss_fn, config, sofa):
        """
        Initialize a trainer with model and training configuration specialized for HRTF estimation.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Neural network model (e.g., PINN for acoustic scattering)
        dataloader : torch.utils.data.DataLoader
            Data loader providing training batches
        loss_fn : callable
            Loss function for training optimization
        config : dict
            Configuration dictionary containing training hyperparameters:
            - 'model': str, model name for logging and checkpointing
            - 'preload': bool, whether to load pre-trained weights
            - other training-specific parameters
        """
        config['hrtf'] = True
        config['custom_shape'] = True
        super().__init__(model, dataloader, loss_fn, config)
        self.sofa = sofa

    def train(self):
        name = self.config['model']
        id = self.config['id']
        log_dir = f'./runs/{name}{id}'
        self.writer = SummaryWriter(log_dir = log_dir)
        print(f'TensorBoard logs are being saved to: {os.path.abspath(log_dir)}')
        exp_dir = f'checkpoints/hrtf/{id}/'
        os.makedirs(exp_dir, exist_ok=True)

        if not self.config['preload']:
            self.train_scattering()
            print(f'Pre-training saved under : {exp_dir}{name}_pre.pth')
            torch.save(self.model.state_dict(), f'{exp_dir}{name}_pre.pth')
            with open(f'{exp_dir}config.yaml', 'w') as f:
                yaml.dump(self.config, f)
        self.fine_tune_scattering()
        print(f'Training saved under : {exp_dir}{name}.pth')
        torch.save(self.model.state_dict(), f'{exp_dir}{name}.pth')
        with open(f'{exp_dir}config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        self.writer.close()
    
    def train_scattering(self):
        config = self.config['adam']
        dataloader_iter = iter(self.dataloader['adam']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])           
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)
        for epoch in range(config['epochs']):
            if epoch%config['keeping_time'] == 0: 
                if epoch != 0:
                    # Only delete if they were defined in a previous epoch
                    del x_sample
                    del boundary_points
                    del normals_points
                    torch.cuda.empty_cache()  # optional but helpful
        
                x_sample = next(dataloader_iter).to(self.device)
                boundary_points, normals_points = sample_with_blue_noise(boundary, normals, config, epoch)
            u, x = self.model(x_sample, diff = True)
            
            pml_constraint = self.pml_3D(u,x)

            loss_real =  torch.abs(pml_constraint[:, 0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:, 1]**2).mean()

            loss_obstacle = self.bc_loss(boundary_points, normals_points)     

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.writer.add_scalar('Loss/real', loss_real.item(), epoch)
            self.writer.add_scalar('Loss/imag', loss_imag.item(), epoch)
            self.writer.add_scalar('Loss/total', loss.item(), epoch)
            self.writer.add_scalar('Loss/obstacle', loss_obstacle.item(), epoch)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                print(f"Epoch {epoch}, Loss real: {loss_real.item():.6f}")
                print(f"Epoch {epoch}, Loss imag: {loss_imag.item():.6f}")
                print(f"Epoch {epoch}, Loss obstacle: {loss_obstacle.item():.6f}")
                print(f"Epoch {epoch}, Grad {torch.linalg.norm(x.grad)}")
            del u
            del x
            if epoch % 2000 == 0:
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch)

            if epoch%10000 == 0 and epoch!=0:
                id = self.config['id']
                name = self.config['model']
                exp_dir = f'checkpoints/hrtf/{id}/'
                print(f'Pre-training saved under : {exp_dir}{name}_pre.pth')
                torch.save(self.model.state_dict(), f'{exp_dir}{name}_pre.pth')
                with open(f'{exp_dir}config.yaml', 'w') as f:
                    yaml.dump(self.config, f)

            self.state = {
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }

    def fine_tune_scattering(self):
        config = self.config['fine']
        dataloader_iter = iter(self.dataloader['fine']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
 
        optimizer = LBFGS(self.model.parameters(), lr=config['lr'], max_iter = config['max_iter'], line_search_fn = 'strong_wolfe')      
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)

        def criterion(x_sample, boundary_points, normals_points):
            u, x = self.model(x_sample, diff = True)
            pml_constraint = self.pml_3D(u,x)
            loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
            loss_obstacle = self.bc_loss(boundary_points, normals_points)   

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle

            del u
            del x
            return loss

        def make_closure(x_sample, boundary_points, normals_points):
            def closure():
                optimizer.zero_grad()
                loss = criterion(x_sample, boundary_points, normals_points)
                loss.backward()
                return loss
            return closure
        
        for epoch in range(config['epochs']):
            self.model.train()
            if epoch%config['keeping_time'] == 0: 
                if epoch!=0:
                    del x_sample
                    torch.cuda.empty_cache()  # optional but helpful
                x_sample = next(dataloader_iter).to(self.device)

            if epoch%config['keeping_time'] == 0: 
                if epoch != 0:
                    # Only delete if they were defined in a previous epoch
                    del boundary_points
                    del normals_points
                boundary_points, normals_points = sample_with_blue_noise(boundary, normals, config, epoch)
                closure = make_closure(x_sample, boundary_points, normals_points)
            loss = optimizer.step(closure)
            scheduler.step()

            self.writer.add_scalar('Loss/total_LBFGS', loss.item(), epoch)

            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % 20 == 0:
                #print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, self.config['adam']['epochs'] + epoch)

            if epoch%10 == 0 and epoch!=0:
                id = self.config['id']
                name = self.config['model']
                exp_dir = f'checkpoints/hrtf/{id}/'
                print(f'Pre-training saved under : {exp_dir}{name}.pth')
                torch.save(self.model.state_dict(), f'{exp_dir}{name}.pth')
                with open(f'{exp_dir}config.yaml', 'w') as f:
                    yaml.dump(self.config, f)
            self.state_fine = {
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }


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

        u_rez = u_real[format]
        u_imz = u_imag[format]
        u_sc_mag = np.sqrt(u_rez**2 + u_imz**2)
        u_full = u+u_inc
        u_full_mag = np.linalg.norm(u_full, axis = 1).reshape(self.res, self.res, self.res)
        u_full_mag = u_full_mag[format]
        u_sc_phase = np.angle(u_rez + 1j * u_imz)
        u_full_phase = np.angle(u_full[:, 0] + 1j * u_full[:, 1]).reshape(self.res, self.res, self.res)
        u_full_phase = u_full_phase[format]

        u_rey = u_real[: , z_choice ,:]
        u_imy = u_imag[: , z_choice ,:]

        u_rex = u_real[z_choice , :, :]
        u_imx = u_imag[z_choice , :, :]

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

        u_sc_mag = np.ma.masked_where(xy_mask, u_sc_mag)
        u_full_mag = np.ma.masked_where(xy_mask, u_full_mag)
        u_sc_phase = np.ma.masked_where(xy_mask, u_sc_phase)
        u_full_phase = np.ma.masked_where(xy_mask, u_full_phase)

        fig, axs = plt.subplots(3, 2, figsize=(12, 5))
        # Real part plot
        im0 = axs[0, 0].imshow(u_rez, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('XY Real part')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0,1].imshow(u_imz, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[0,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0,1].set_title('XY Imag part')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part plot
        im2 = axs[1, 0].imshow(u_rey , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('XZ Real part')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part plot
        im3 = axs[1,1].imshow(u_imy , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[1,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1,1].set_title('XZ Imag part')
        fig.colorbar(im3, ax=axs[1, 1])
        plt.tight_layout()

        # Real part plot
        im4 = axs[2, 0].imshow(u_rex , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[2, 0].set_title('YZ Real part')
        fig.colorbar(im4, ax=axs[2, 0])

        # Imaginary part plot
        im5 = axs[2,1].imshow(u_imx , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[1,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
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
        #self.writer.add_image(f'Scattered_field_pred/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard
        self.writer.add_image(f'Scattered_field_HRTF/{epoch}', image_tensor[0], 0)


        # Close the plot and the buffer
        plt.close()
        buf.close()

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Scattered field magnitude (top-left)
        im0 = axs[0, 0].imshow(u_sc_mag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Scattered field magnitude')
        fig.colorbar(im0, ax=axs[0, 0])

        # Full field magnitude (top-right)
        im1 = axs[0, 1].imshow(u_full_mag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='turbo')
        #axs[0, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 1].set_title('Full field magnitude')
        fig.colorbar(im1, ax=axs[0, 1])

        # Scattered field phase (bottom-left)
        im2 = axs[1, 0].imshow(u_sc_phase, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('Scattered field phase')
        fig.colorbar(im2, ax=axs[1, 0], label='radians')

        # Full field phase (bottom-right)
        im3 = axs[1, 1].imshow(u_full_phase, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        #axs[1, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 1].set_title('Full field phase')
        fig.colorbar(im3, ax=axs[1, 1], label='radians')

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

        k = self.k.item() 
        res_pde_real = np.abs(lap_u_real - k ** 2 * self.config['scale'] ** 2 * u_real) + 1e-6 
        res_pde_imag = np.abs(lap_u_imag - k ** 2 * self.config['scale'] ** 2 * u_imag) + 1e-6

        res_pde_real = np.log(res_pde_real)[:, :, z_choice]
        res_pde_imag = np.log(res_pde_imag)[:, :, z_choice]

        res_pde_real = np.ma.masked_where(xy_mask, res_pde_real)
        res_pde_imag = np.ma.masked_where(xy_mask, res_pde_imag)
        # Real part plot
        im0 = axs[0].imshow(res_pde_real, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='plasma')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0].set_title('Residual of real part')
        fig.colorbar(im0, ax=axs[0])

        # Imaginary part plot
        im1 = axs[1].imshow(res_pde_imag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='plasma')
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
