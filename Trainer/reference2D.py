from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import save_lora_weights
from visuals import create_obstacle_patch
from torch.utils.tensorboard import SummaryWriter
import time
from .base import BaseTrainer2D
from io import BytesIO
from PIL import Image

class Trainer2D(BaseTrainer2D):
    def __init__(self, model, dataloader, loss_fn, config):
        # Call parent constructor (BaseTrainer2D.__init__)
        super().__init__(model, dataloader, loss_fn, config)

    def train(self, save_dir = 'checkpoints/2D/scattering/', lora_dir = 'checkpoints/2D/lora/'):
        """
        Train the model in function of the desired training :
        If self.lora is true, it means that we try to adapt a reference model to another direction, so only LoRA training is done.
        If self.preload is true, it means that we have partly trained our model and want to perform only L BFGS training phase.
        """
        start = time.time()
        name = self.config['model']
        log_dir = f'./runs/{name}'
        self.writer = SummaryWriter(log_dir = log_dir)
        self.save_dir = save_dir
        print(f'TensorBoard logs are being saved to: {os.path.abspath(log_dir)}')
        if self.lora : 
            print('Changing direction with LoRA')
            self.train_lora()
            # Convert direction to filename format
            direction_str = ""
            for coord in self.direction.squeeze(1):
                if coord >= 0:
                    direction_str += f"+{coord}"
                else:
                    direction_str += f"{coord}"  # negative sign is already included

            filename = f'{lora_dir}{direction_str}.pth'
            save_lora_weights(self.model, filename)
            print(f'LoRA weights saved under : {filename}')
        else : 
            if not self.config['preload']:
                self.train_scattering()
                mid = time.time()
                pre_training_time = mid - start
                print('Pretraining time :', pre_training_time)
                print(f'Pre-training saved under : {save_dir}{name}_pre.pth')
                torch.save(self.model.state_dict(), f'{save_dir}{name}_pre.pth')
            self.fine_tune_scattering()
            print(f'Training saved under : {save_dir}{name}.pth')
            torch.save(self.model.state_dict(), f'{save_dir}{name}.pth')
        end = time.time()
        training_time = end - start
        if self.lora:
            print('LoRA adaptation training time : ', training_time)
        elif self.config['preload']:
            print('Training time with preloaded weights : ', training_time)
        else :
            print('Full training time : ', training_time)
        self.writer.close()


    def train_scattering(self):
        config = self.config['adam']
        dataloader_iter = iter(self.dataloader['adam']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])   
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)

        for epoch in range(config['epochs']):
            self.model.train()
            if epoch%config['keeping_time'] == 0: 
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_size'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]

            u, x = self.model(x_sample, diff = True)
            pml_constraint = self.pml_2d(u,x)
            loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
            loss_obstacle = self.bc_loss(boundary_points, normals_points)     #SAME DIMENSION AS WAVE EQUATION BECAUSE IT IS FAR FROM FAKE BOUNDARY

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


            self.writer.add_scalar('Loss/real', loss_real.item(), epoch)
            self.writer.add_scalar('Loss/imag', loss_imag.item(), epoch)
            self.writer.add_scalar('Loss/total', loss.item(), epoch)
            self.writer.add_scalar('Loss/obstacle', loss_obstacle.item(), epoch)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % 2000 == 0:
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch)
                save_dir = self.save_dir
                name = self.config['model']
                torch.save(self.model.state_dict(), f'{save_dir}{name}_pre.pth')


    def fine_tune_scattering(self):
        config = self.config['fine']
        dataloader_iter = iter(self.dataloader['fine']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        optimizer = torch.optim.LBFGS(list(self.model.parameters()), lr=config['lr'], max_iter = config['max_iter'], line_search_fn = 'strong_wolfe')     

        def criterion(x_sample, boundary_points, normals_points):
            u, x = self.model(x_sample, diff = True)
            pml_constraint = self.pml_2d(u,x)
            loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
            loss_obstacle = self.bc_loss(boundary_points, normals_points)

            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
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
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_boundary'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]
                closure = make_closure(x_sample, boundary_points, normals_points)

            loss = optimizer.step(closure)
            self.writer.add_scalar('Loss/total', loss.item(), epoch+self.config['adam']['epochs'])

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % 25 == 0:
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch+self.config['adam']['epochs'])
                save_dir = self.save_dir
                name = self.config['model']
                torch.save(self.model.state_dict(), f'{save_dir}{name}.pth')

            self.state_fine = {
                'optimizer' : optimizer
            }


    def train_lora(self):
        config = self.config['lora']
        dataloader_iter = iter(self.dataloader['lora']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        optimizer = config['optimizer'] 
        
        def criterion(x_sample, boundary_points, normals_points):
            u, x = self.model(x_sample, diff = True)
            pml_constraint = self.pml_2d(u,x)
            loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
            loss_obstacle = self.bc_loss(boundary_points, normals_points)
            loss = self.weight_re * loss_real + self.weight_im * loss_imag + self.weight_bc * loss_obstacle
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
                x_sample = next(dataloader_iter).to(self.device)
                idx_bd = torch.randint(0, boundary.size(0), (config['batch_boundary'],), device=boundary.device) 
                boundary_points, normals_points = boundary[idx_bd], normals[idx_bd]
                closure = make_closure(x_sample, boundary_points, normals_points)
            loss = optimizer.step(closure)
            self.writer.add_scalar('Loss/lora', loss.item(), epoch)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % 25 == 0:
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch+self.config['adam']['epochs'])

            self.state_fine = {
                'optimizer' : optimizer
                }
            


    def save_plot(self, u, u_inc, epoch):
        # Create the figure and subplots
        u_real = u[:, 0].reshape(self.res, self.res)
        u_imag = u[:, 1].reshape(self.res, self.res)
        
        # Handle both single polygon and multiple polygons
        polygon_data = self.dataloader['polygon']
        if isinstance(polygon_data, list):
            # Multiple polygons - each needs to be flipped
            polygons = [poly[:, ::-1] for poly in polygon_data]
        else:
            # Single polygon - flip and convert to list for uniform handling
            polygons = [polygon_data[:, ::-1]]

        target = np.zeros((self.res, self.res, 2))
        target[..., 0] = self.tar[:, 0].cpu().numpy().reshape(self.res, self.res)
        target[..., 1] = self.tar[:, 1].cpu().numpy().reshape(self.res, self.res)
        du_r = (target[1:, :, 0] - target[:-1, :, 0]) / self.res
        du_i = (target[1:, :, 1] - target[:-1, :, 1]) / self.res

        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        
        # Real part plot
        im0 = axs[0, 0].imshow(u_real, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        # Add all polygon patches
        for polygon in polygons:
            axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Scattered field real part')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0, 1].imshow(u_imag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[0, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 1].set_title('Scattered field imag part')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part target/difference plot
        im2 = axs[1, 0].imshow(target[..., 0], extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('Difference with groundtruth real part')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part target/difference plot
        im3 = axs[1, 1].imshow(target[..., 1], extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[1, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 1].set_title('Difference with groundtruth imag part')
        fig.colorbar(im3, ax=axs[1, 1])

        #plt.subplots_adjust(wspace=0.1, hspace=0.3)

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

        # Second figure - Full field comparison
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        u_sc = np.sqrt(u_real**2 + u_imag**2)
        u_full = np.linalg.norm(u + u_inc, axis=1).reshape(self.res, self.res)
        u_full_gt = np.linalg.norm(self.tar.cpu().numpy() + u_inc, axis=1).reshape(self.res, self.res)
        
        # Ground truth full field plot
        im0 = axs[0].imshow(u_sc, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0].set_title('Scattered field')
        fig.colorbar(im0, ax=axs[0])

        # Predicted full field plot
        im1 = axs[1].imshow(u_full, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
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

        # Third figure - Derivative comparison
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))

        du_real = (u_real[1:] - u_real[:-1]) * self.res
        du_imag = (u_imag[1:] - u_imag[:-1]) * self.res

        u = u.reshape(self.res, self.res, 2)

        # Real part of estimated derivative
        im0 = axs[0, 0].imshow(np.log(np.abs(du_real) + 1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Real part of dP_s estimated')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part of estimated derivative
        im1 = axs[0, 1].imshow(np.log(np.abs(du_imag) + 1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[0, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 1].set_title('Imag part of dP_s estimated')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part of true derivative
        im2 = axs[1, 0].imshow(np.log(np.abs(du_r) + 1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
            axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('Real part of  dP_s true')
        fig.colorbar(im2, ax=axs[1, 0])
                        
        # Imaginary part of true derivative
        im3 = axs[1, 1].imshow(np.log(np.abs(du_i) + 1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='jet')
        for polygon in polygons:
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
