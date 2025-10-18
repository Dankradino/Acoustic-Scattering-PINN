from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import generate_grid, save_lora_weights
from visuals import create_obstacle_patch
from eval import sound_hard_circle
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
from PIL import Image
import time
from .base import BaseTrainer2D


class ConditionedTrainer2D(BaseTrainer2D):
    def __init__(self, model, dataloader, loss_fn, config):
        """
        Initialize 2D trainer with model and training configuration.
        This trainer is specialized for the conditioned PINN using two input :
            -coordinate
            -direction
        
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
        super().__init__(model, dataloader, loss_fn, config)
        self.init_solution()

    def create_forward_method(self, direction):
        """Create forward method with captured parameters"""
        def forward_method(coords):
            return self.model(coords, direction, diff=True)
        return forward_method
    
    def train(self, save_dir = 'checkpoints/conditioned/'):
        start = time.time()
        name = self.config['model']
        log_dir = f'./runs/conditioned_{name}'
        self.writer = SummaryWriter(log_dir = log_dir)
        self.save_dir = save_dir
        print(f'TensorBoard logs are being saved to: {os.path.abspath(log_dir)}')
        if not self.config['preload']:
            self.train_scattering()
            print(f'Pre-training saved under : {save_dir}{name}_pre_conditioned.pth')
            torch.save(self.model.state_dict(), f'{save_dir}{name}_pre_conditioned.pth')

        print('Training direction')
        self.train_direction()
        print(f'Pre-training saved under : {save_dir}{name}_pre_conditioned.pth')
        torch.save(self.model.state_dict(), f'{save_dir}{name}_pre_conditioned.pth')
        self.fine_tune_scattering_over_direction()
        print(f'Training saved under : {save_dir}{name}_conditioned.pth')
        torch.save(self.model.state_dict(), f'{save_dir}{name}_conditioned.pth')
        end = time.time()
        training_time = end - start
        if self.config['preload']:
            print('Training time with preloaded weights : ', training_time)
        else :
            print('Full training time : ', training_time)
        self.writer.close()

    def train_scattering(self):
        config = self.config['adam']
        dataloader_iter = iter(self.dataloader['adam']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)
        forward_method = self.create_forward_method(self.direction.T)
        for epoch in range(config['epochs']):
            self.model.train()
            if epoch%config['keeping_time'] == 0: 
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]

            u, x = self.model(x_sample, self.direction.T, diff = True)
            pml_constraint = self.pml_2d(u,x)
            loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
            loss_obstacle = self.bc_loss(boundary_points, normals_points, forward_method = forward_method)   

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.writer.add_scalar('Loss/real', np.log(loss_real.item()), epoch)
            self.writer.add_scalar('Loss/imag', np.log(loss_imag.item()), epoch)
            self.writer.add_scalar('Loss/total', loss.item(), epoch)
            self.writer.add_scalar('Loss/obstacle', np.log(loss_obstacle.item()), epoch)


            if epoch % 250 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                print(f"Epoch {epoch}, Grad {torch.linalg.norm(x.grad)}")

            if epoch % 2000 == 0:
                #print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                with torch.no_grad():
                    u = self.model(self.x_grid, self.direction.T).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, self.direction).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch)

            self.state = {
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }

    def fine_tune_scattering_over_direction(self):
        config = self.config['fine']
        dataloader_iter = iter(self.dataloader['fine']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=config['lr'], max_iter = config['max_iter'], line_search_fn = 'strong_wolfe')  
 
        def criterion(x_sample, boundary_points, normals_points, direction_batch):
            loss_real = 0
            loss_imag = 0
            loss_obstacle = 0
            for direction in direction_batch:
                direction = direction.unsqueeze(0)
                u, x = self.model(x_sample, direction, diff = True)
                pml_constraint = self.pml_2d(u, x)
                loss_real +=  torch.abs(pml_constraint[:,0]**2).mean()
                loss_imag +=  torch.abs(pml_constraint[:,1]**2).mean()

                forward_method = self.create_forward_method(direction)
                loss_obstacle += self.bc_loss(boundary_points, normals_points, direction = direction[0], forward_method = forward_method)  

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
            return loss

        def make_closure(x_sample, boundary_points, normals_points, direction_batch):
            def closure():
                optimizer.zero_grad()
                loss = criterion(x_sample, boundary_points, normals_points, direction_batch)
                loss.backward()
                return loss
            return closure
        self.model.train()
        direction_keeping = config['direction_keeping_time']
        for epoch in range(config['epochs']):
            if epoch%config['keeping_time'] == 0: 
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]

            if epoch % direction_keeping == 0:
                direction_batch = self._sample_random_directions(config['batch_direction'])
                closure = make_closure(x_sample, boundary_points, normals_points, direction_batch)

            loss = optimizer.step(closure)

            self.writer.add_scalar('Loss/total', loss.item(), epoch+self.config['adam']['epochs'])

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % 25 == 0:
                with torch.no_grad():
                    u = self.model(self.x_grid, direction_batch[-1]).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, direction_batch[-1].T).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch+self.config['adam']['epochs'])

            if epoch%100 == 0:
                name = self.config['model']
                save_dir = self.save_dir
                torch.save(self.model.state_dict(), f'{save_dir}{name}_conditioned.pth')
            self.state_fine = {
                'optimizer' : optimizer
            }
    
    def train_direction(self):
        config = self.config['adam']
        dataloader_iter = iter(self.dataloader['adam']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)
        self.model.train()
        direction_keeping = config['direction_keeping_time']
        for epoch in range(config['epochs']):
            if epoch%config['keeping_time'] == 0: 
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]

            if epoch % direction_keeping == 0:
                direction_batch = self._sample_random_directions(config['batch_direction'])

            loss_real = 0
            loss_imag = 0
            loss_obstacle = 0
            for direction in direction_batch:
                direction = direction.unsqueeze(0)
                u, x = self.model(x_sample, direction, diff = True)
                pml_constraint = self.pml_2d(u,x)
                loss_real +=  torch.abs(pml_constraint[:,0]**2).mean()
                loss_imag +=  torch.abs(pml_constraint[:,1]**2).mean()
                forward_method = self.create_forward_method(direction)
                loss_obstacle += self.bc_loss(boundary_points, normals_points, direction = direction[0], forward_method = forward_method)     #SAME DIMENSION AS WAVE EQUATION BECAUSE IT IS FAR FROM FAKE BOUNDAR

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


            self.writer.add_scalar('Loss/real', np.log(loss_real.item()), epoch)
            self.writer.add_scalar('Loss/imag', np.log(loss_imag.item()), epoch)
            self.writer.add_scalar('Loss/total', loss.item(), epoch)
            self.writer.add_scalar('Loss/obstacle', np.log(loss_obstacle.item()), epoch)

            if epoch % 250 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                print(f"Epoch {epoch}, Grad {torch.linalg.norm(x.grad)}")
            if epoch % 1000 == 0:
                with torch.no_grad():
                    print('direction', direction)
                    self.config['direction'] = direction.T
                    u = self.model(self.x_grid, direction).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid, direction.T).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch)

            if epoch%1000 == 0:
                name = self.config['model']
                save_dir = self.save_dir
                torch.save(self.model.state_dict(), f'{save_dir}{name}_pre_conditioned.pth')

            self.state = {
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }

    

    def _sample_random_directions(self, batch_size, ratio = 1.):
        """Sample random directions on unit circle"""
        angles = torch.linspace(0, 2*np.pi, batch_size, device = self.device) + ratio * torch.randn(batch_size, device = self.device)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return directions

    def save_plot(self, u, u_inc, epoch):
        # Create the figure and subplots
        u_real = u[:, 0].reshape(self.res, self.res)
        u_imag = u[:, 1].reshape(self.res, self.res)
        polygon = self.dataloader['polygon']

        R = 0.5
        mask = self.x_grid[:,0]**2 + self.x_grid[:,1]**2 > R**2
        #print('mask shape :' ,mask.shape)
        tar = torch.zeros((self.res**2,2), device = self.device, dtype = torch.double)
        tar[mask,:] = sound_hard_circle(self.config, self.x_grid[mask,:], R)
        target = np.zeros((self.res,self.res,2))
        target[...,0] = tar[:,0].cpu().numpy().reshape(self.res,self.res)
        target[...,1] = tar[:,1].cpu().numpy().reshape(self.res,self.res)
        du_r = (target[1:,:,0]-target[:-1,:,0])/self.res
        du_i = (target[1:,:,1]-target[:-1,:,1])/self.res



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
        # print("u shape:", u.shape)
        # print("u_inc shape:", u_inc.shape)
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

        du_real = (u_real[1:]-u_real[:-1]) * self.res
        du_imag = (u_imag[1:]-u_imag[:-1]) * self.res

        u = u.reshape(self.res,self.res,2)

        # Real part plot
        im0 = axs[0, 0].imshow(np.log(np.abs(du_real)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Real part of dP_s estimated')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0, 1].imshow(np.log(np.abs(du_imag)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[0, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 1].set_title('Imag part of dP_s estimated')
        fig.colorbar(im1, ax=axs[0, 1])

        im2 = axs[1, 0].imshow(np.log(np.abs(du_r)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('Real part of  dP_s true')
        fig.colorbar(im2, ax=axs[1, 0])
                     
        im3 = axs[1, 1].imshow(np.log(np.abs(du_i)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        axs[1, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 1].set_title('Imag part of dP_s true')
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
        self.writer.add_image(f'dP_s_comparison/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard

        # Close the plot and the buffer
        plt.close()
        buf.close()
