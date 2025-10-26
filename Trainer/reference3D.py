from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from utils import save_lora_weights, generate_grid
from io import BytesIO
from PIL import Image
import time
from utils import sample_with_blue_noise
from model.lora import * 
from .base import BaseTrainer3D


class Trainer3D(BaseTrainer3D):
    def __init__(self, model, dataloader, loss_fn, config):
        config['hrtf'] = False
        super().__init__(model, dataloader, loss_fn, config)
        
    def train(self, save_dir = 'checkpoints/3D/scattering/', lora_dir = 'checkpoints/3D/lora/'):
        start = time.time()
        name = self.config['model']
        log_dir = f'./runs/{name}'
        self.save_dir = save_dir
        self.lora_dir = lora_dir
        self.writer = SummaryWriter(log_dir = log_dir)
        print(f'TensorBoard logs are being saved to: {os.path.abspath(log_dir)}')
        if self.lora : 
            # Convert direction to filename format
            direction_str = ""

            for coord in self.direction.squeeze(1):
                if coord >= 0:
                    direction_str += f"+{coord}"
                else:
                    direction_str += f"{coord}"  # negative sign is already included

            checkpoint_path = f'{lora_dir}{direction_str}.pth'

            print('Changing direction with LoRA')
            self.train_lora()
            save_lora_weights(self.model, checkpoint_path)
            print(f'LoRA weights saved under : {checkpoint_path}')

        else:
            if not self.config['preload']:
                self.train_scattering()
                print(f'Pre-training saved under : {save_dir}{name}_pre.pth')
                torch.save(self.model.state_dict(), f'{save_dir}{name}_pre.pth')
                mid = time.time()
                pre_training_time = mid - start
                print('Pretraining time :', pre_training_time)
            self.fine_tune_scattering()
            print(f'Training saved under : {save_dir}{name}.pth')
            torch.save(self.model.state_dict(), f'{save_dir}{name}.pth')
        end = time.time()
        training_time = end - start
        print('Training time with preload as',self.config['preload'], ' :', training_time)
        self.writer.close()
    
    def train_scattering(self):
        config = self.config['adam']
        dataloader_iter = iter(self.dataloader['adam']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=config['lr']) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)

        for epoch in range(config['epochs']):
            self.model.train()
            if epoch%config['keeping_time'] == 0: 
                if epoch != 0:
                    # Only delete if they were defined in a previous epoch
                    del boundary_points
                    del normals_points
                x_sample = next(dataloader_iter).to(self.device)
                boundary_points, normals_points = sample_with_blue_noise(boundary, normals, config, epoch)

            u, x = self.model(x_sample, diff = True)
            pml_constraint = self.pml_3D(u,x)
            loss_real =  torch.abs(pml_constraint[:,0]**2).mean()
            loss_imag =  torch.abs(pml_constraint[:,1]**2).mean()
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

            if epoch % 250 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                print(f"Epoch {epoch}, Grad {torch.linalg.norm(x.grad)}")
            if epoch % 2000 == 0:
                #print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch)
                save_dir = self.save_dir
                name = self.config['model']
                torch.save(self.model.state_dict(), f'{save_dir}{name}_pre.pth')

            self.state = {
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }

    def fine_tune_scattering(self):
        config = self.config['fine']
        dataloader_iter = iter(self.dataloader['fine']) 
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=config['lr'], max_iter = config['max_iter'], line_search_fn = 'strong_wolfe')   
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[config['n_scheduler']*(i+1) for i in range(config['epochs']//config['n_scheduler'] - 1)],gamma=config['gamma'],)

        
        def criterion(x_sample, boundary_points, normals_points):
            u, x = self.model(x_sample, diff = True)
            pml_constraint = self.pml_3D(u,x)
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
                if epoch != 0:
                    # Only delete if they were defined in a previous epoch
                    del boundary_points
                    del normals_points
                x_sample = next(dataloader_iter).to(self.device)
                boundary_points, normals_points = sample_with_blue_noise(boundary, normals, config, epoch)
                closure = make_closure(x_sample, boundary_points, normals_points)
            loss = optimizer.step(closure)
            scheduler.step()

            self.writer.add_scalar('Loss/total', loss.item(), epoch+self.config['adam']['epochs'])

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % 25 == 0:
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch + self.config['adam']['epochs'])

            if epoch%10 == 0 and epoch!=0:
                save_dir = self.save_dir
                name = self.config['model']
                torch.save(self.model.state_dict(), f'{save_dir}{name}_pre.pth')


            self.state_fine = {
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }
    

    def train_lora(self):
        config = self.config['lora']   
        dataloader_iter = iter(self.dataloader['lora'])  
        printer = 5
        plotter = 25
        boundary = self.dataloader['boundary']
        normals = self.dataloader['normals']
        optimizer = config['optimizer'] 

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
            
            if epoch%(config['keeping_time']) == 0:
                if epoch != 0:
                    # Only delete if they were defined in a previous epoch
                    del boundary_points
                    del normals_points
                    
                boundary_points, normals_points = sample_with_blue_noise(boundary, normals, config, epoch)
                closure = make_closure(x_sample, boundary_points, normals_points)
            loss = optimizer.step(closure)

            self.writer.add_scalar('Loss/lora', loss.item(), epoch)

            if epoch % printer == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if epoch % plotter == 0:
                #print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                with torch.no_grad():
                    u = self.model(self.x_grid).cpu().numpy()
                    u_inc = self.inc_wave(self.x_grid).squeeze(1).cpu().numpy()
                self.save_plot(u, u_inc, epoch+self.config['adam']['epochs'])
                direction_str = ""

                for coord in self.direction.squeeze(1):
                    if coord >= 0:
                        direction_str += f"+{coord}"
                    else:
                        direction_str += f"{coord}"  # negative sign is already included
                lora_dir = self.lora_dir
                filename = f'{lora_dir}{direction_str}.pth'
                save_lora_weights(self.model, filename)
                print(f'LoRA weights saved under : {filename}')

            self.state_fine = {
                'optimizer' : optimizer
                }

    def save_plot(self, u, u_inc, epoch):
        # Create the figure and subplots
        z_choice = self.res//2
        format = (slice(None), slice(None), z_choice)
        u_real = u[:, 0].reshape(self.res, self.res, self.res)
        u_imag = u[:, 1].reshape(self.res, self.res, self.res)
        u_real[self.dataloader['mesh_mask']] = 0
        u_imag[self.dataloader['mesh_mask']] = 0
        u_real = u_real[format]
        u_imag = u_imag[format]
        u_sc = np.sqrt(u_real**2+u_imag**2)
        # print("u shape:", u.shape)
        # print("u_inc shape:", u_inc.shape)
        u_full = np.linalg.norm(u + u_inc, axis = 1).reshape(self.res, self.res, self.res)
        u_full[self.dataloader['mesh_mask']] = 0.
        u_full = u_full[format]
        target = np.zeros((self.res,self.res,2))
        target[...,0] = u_full#self.reshape_solution[format + (0,)] 
        target[...,1] = u_full#self.reshape_solution[format + (1,)] 
        du_r = (target[1:,:,0]-target[:-1,:,0])/self.res
        du_i = (target[1:,:,1]-target[:-1,:,1])/self.res

        diff_real = u[:, 0] #np.abs(u[:, 0] - self.solution[:, 0])
        diff_real = diff_real.reshape(self.res, self.res, self.res)
        diff_imag = u[:, 1] #np.abs(u[:, 1] - self.solution[:, 1])
        diff_imag = diff_imag.reshape(self.res, self.res, self.res)
        u_rez = diff_real[format]
        u_imz = diff_imag[format]
        u_rey = diff_real[: , z_choice ,:]
        u_imy = diff_imag[: , z_choice ,:]
        u_rex = diff_real[z_choice , :, :]
        u_imx = diff_imag[z_choice , :, :]

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


        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        # Real part plot
        im0 = axs[0, 0].imshow(u_real, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Scattered field real part')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0,1].imshow(u_imag, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0,1].set_title('Scattered field imag part')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part plot
        im2 = axs[1, 0].imshow(target[...,0] , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('True scattered field real part')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part plot
        im3 = axs[1,1].imshow(target[...,1] , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1,1].set_title('True scattered field imag part')
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
        #self.writer.add_image(f'Scattered_field_pred/{epoch}', image_tensor[0], 0)  # Add the image to TensorBoard
        self.writer.add_image(f'Scattered_field_pred/{epoch}', image_tensor[0], 0)


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
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('XZ Real part')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part plot
        im3 = axs[1,1].imshow(u_imy , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1,1].set_title('XZ Imag part')
        fig.colorbar(im3, ax=axs[1, 1])
        plt.tight_layout()

        # Real part plot
        im4 = axs[2, 0].imshow(u_rex , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[2, 0].set_title('YZ Real part')
        fig.colorbar(im4, ax=axs[2, 0])

        # Imaginary part plot
        im5 = axs[2,1].imshow(u_imx , extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
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
        self.writer.add_image(f'Scattered_field/{epoch}', image_tensor[0], 0)


        # Close the plot and the buffer
        plt.close()
        buf.close()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Real part plot
        im0 = axs[0].imshow(u_sc, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0].set_title('Scattered field')
        fig.colorbar(im0, ax=axs[0])

        # Imaginary part plot
        im1 = axs[1].imshow(u_full, extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
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

        # Real part plot
        im0 = axs[0, 0].imshow(np.log(np.abs(du_real)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Real part of dP_s estimated')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0, 1].imshow(np.log(np.abs(du_imag)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[0, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 1].set_title('Imag part of dP_s estimated')
        fig.colorbar(im1, ax=axs[0, 1])

        im2 = axs[1, 0].imshow(np.log(np.abs(du_r)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[1, 0].set_title('Real part of  dP_s true')
        fig.colorbar(im2, ax=axs[1, 0])
                     
        im3 = axs[1, 1].imshow(np.log(np.abs(du_i)+1e-10), extent=[-self.L, self.L, -self.L, self.L], origin='lower', cmap='viridis')
        #axs[1, 1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
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



