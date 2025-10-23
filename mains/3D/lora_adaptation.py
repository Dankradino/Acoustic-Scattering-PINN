import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import torch
import torch.nn as nn
import numpy as np
from model import init_model_from_conf, init_with_Lora, init_with_Lora_rff
from shape import generate_sphere, get_sphere_param
from utils import create_3d_mesh_mask
from Trainer import Trainer3D
from Dataloader import create_dataloader3D, loader3D
from eval import evaluate_sphere_estimation
import yaml
import trimesh
import gc

def fibonacci_sphere(n_points, device='cpu', dtype=torch.float32):
    """Generate approximately evenly distributed unit vectors on a sphere."""
    indices = torch.arange(0, n_points, dtype=dtype, device=device) + 0.5

    phi = torch.acos(1 - 2 * indices / n_points)       # polar angle
    theta = np.pi * (1 + 5**0.5) * indices             # golden angle * index

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    directions = torch.stack([x, y, z], dim=1)  # (n_points, 3)
    return directions


def train_direction(i, direction, config, model_name, dataloader_cpu, loss_fn, mesh_param, save_dir, lora_dir):
        import torch
        from copy import deepcopy
        torch.manual_seed(30 + i)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move direction to GPU and normalize
        direction = direction.to(device)
        direction =  direction / torch.linalg.norm(direction)

        print('direction' , direction)

        conf_copy = deepcopy(config)
        conf_copy['direction'] = direction

        # Move dataloader tensors to GPU
        dataloader = {}
        for key, value in dataloader_cpu.items():
            if isinstance(value, torch.Tensor):
                dataloader[key] = value.to(device)
            else:
                dataloader[key] = value

        model = init_model_from_conf(conf_copy).to(device)
        model.load_state_dict(torch.load(f'{save_dir}{model_name}.pth'))               
        if model_name == 'rff':
            model, optimizer = init_with_Lora_rff(model, conf_copy['lora'], r=12)  #Justification for r=12 is done in the github report.
        else:  
            model, optimizer = init_with_Lora(model, conf_copy['lora'], r=12)
        conf_copy['lora']['optimizer'] = optimizer

        trainer = Trainer3D(model, dataloader, loss_fn, conf_copy)
        trainer.train(lora_dir=lora_dir)
        model.eval()

        evaluate_sphere_estimation(model, trainer, conf_copy, mesh_param['R'])
        del model, trainer, dataloader, conf_copy


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--retrain', type=bool, default=False, help='True if no pretrained model')
    parser.add_argument('--J', type=int, default=6, help='Number of LoRA adapted directions')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/3D/scattering/', help='Save file for weights')
    parser.add_argument('--lora_dir', type=str, default = 'checkpoints/3D/lora/', help='Save file for weights')
    parser.add_argument('--mesh_path', type=str, default=None, help='File containing a custom shape, if None, train on a regular circle defined by mesh_param')
    args = parser.parse_args()
    model_name = args.model
    retrain = args.retrain
    J = args.J
    save_dir = args.save_dir
    lora_dir = args.save_dir
    
    torch.manual_seed(30)

    with open(f"config/scat3D/config_{model_name}.yaml") as file:
        config = yaml.safe_load(file)

    #Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float

    # Grid parameters
    L = config['L']
    Lpml = config['pml_size']
    fake_bd = L + Lpml #Set to 1

    config['preload'] = False
    direction = torch.tensor([1., 1., 1.],device = device).unsqueeze(1)
    direction =  direction / torch.linalg.norm(direction)
    config['direction'] = direction
    mesh_param = {
        'r' : 0.2,
        'center' : (0., 0., 0.),
        'epsilon' : 1e-2,
    }

    loss_fn = nn.MSELoss()
    mesh_path = generate_sphere(radius=mesh_param['R'], subdivisions=8)

    mesh =  trimesh.load(mesh_path)
    if mesh_path is None:
        config['custom_shape'] = False
        mesh_path = generate_sphere(radius=mesh_param['r'], subdivisions=8, center = mesh_param['center'])
        mesh =  trimesh.load(mesh_path)
        boundary , normals = get_sphere_param(mesh.vertices)
    else :
        config['custom_shape'] = True
        # Load almond mesh for trimesh
        mesh =  trimesh.load(mesh_path)
        normals = mesh.vertex_normals
    print(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    
    boundary = torch.tensor(boundary, dtype=DTYPE, device=device)
    normals = torch.tensor(normals, dtype=DTYPE, device=device)

    if retrain : 
        print('Train a model from scratch')
        model = init_model_from_conf(config).to(device)
        create_dataloader3D(
        mesh=mesh,
        cache_dir='./data_points',
        config=config,
        pml_boundary=fake_bd,
        method='spherical_shell'
        )
        points = loader3D(config, cache_dir='./data_points')

        dataloader = {
            'adam' : points['adam'],
            'fine' : points['fine'],
            'normals' : normals,
            'boundary' : boundary,
            'polygon' : mesh.vertices,
            'mesh_mask': create_3d_mesh_mask(config, mesh),
            'R': mesh_param['R']

        }
        trainer = Trainer3D(model, dataloader, loss_fn, config)
        trainer.train(save_dir=save_dir)
        evaluate_sphere_estimation(model, trainer, config, mesh_param['R'])
    else : 
        create_dataloader3D(
        mesh=mesh,
        cache_dir='./data_points',
        config=config,
        pml_boundary=fake_bd,
        method='spherical_shell'
        )
        points = loader3D(config, cache_dir='./data_points')
    dataloader_cpu = {
        'lora' : points['fine'],    #N_points[fine] > N_points[lora] so we ccan reuse the same one, you can regenerate points if you want
        'normals' : normals.cpu(),
        'boundary' : boundary.cpu(),
        'polygon' : mesh.vertices,
        'mesh_mask': create_3d_mesh_mask(config, mesh),
        'R': mesh_param['R']
    }

    config['lora_train'] = True

    directions = fibonacci_sphere(J, device='cpu', dtype=DTYPE) 
    directions = directions.unsqueeze(-1) 

    multiprocess = False # SET TO TRUE IF MULTIPROCESS LEARNING IS POSSIBLE
    if multiprocess : 
        processes = []

        for i, direction in enumerate(directions):
            p = mp.Process(
                target=train_direction,
                args=(i, direction, config, model_name, dataloader_cpu, loss_fn, mesh_param, save_dir, lora_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else :
        for i, direction in enumerate(directions):
            train_direction(
                i, direction, config, model_name, dataloader_cpu, loss_fn, mesh_param, save_dir, lora_dir)
            
            gc.collect()               # clear unused Python objects
            torch.cuda.empty_cache()   # release cached CUDA memory
        
if __name__ == '__main__':
    main()