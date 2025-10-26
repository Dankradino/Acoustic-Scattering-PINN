import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import torch
import torch.nn as nn
from model import init_model_from_conf, init_with_Lora, init_with_Lora_rff
from shape import generate_sphere, get_sphere_param
from Trainer import Trainer3D
from Trainer.phisk3D import initialize_phisk_trainer3D 
from Dataloader import create_dataloader3D, loader3D
from utils import create_3d_mesh_mask, load_config, fibonacci_sphere
from eval import evaluate_sphere_estimation, evaluate_custom_estimation
import yaml
import trimesh
import gc

def train_direction(i, direction, config, model_name, dataloader_cpu, loss_fn, save_dir, lora_dir):
        '''
        Adapt a reference model to a new direction using LoRA scheme and save produced weights in lora_dir
        '''
        import torch
        from copy import deepcopy
        torch.manual_seed(30 + i)

        # Move direction to GPU and normalize
        direction = direction.to(config['device'])
        direction =  direction / torch.linalg.norm(direction)
        print('Adapted direction' , direction)

        conf_copy = deepcopy(config)
        conf_copy['direction'] = direction

        # Move dataloader tensors to GPU
        dataloader = {}
        for key, value in dataloader_cpu.items():
            if isinstance(value, torch.Tensor):
                dataloader[key] = value.to(config['device'])
            else:
                dataloader[key] = value

        #Load reference model
        reference_model = init_model_from_conf(conf_copy).to(config['device'])
        reference_model.load_state_dict(torch.load(f'{save_dir}{model_name}.pth')) 

        #Initialize adapted network              
        if model_name == 'rff':
            model, optimizer = init_with_Lora_rff(reference_model, conf_copy['lora'], r=config['lora_r'], custom_activation=model[1])  #Justification for r=12 is done in the github report.
        else:  
            model, optimizer = init_with_Lora(reference_model, conf_copy['lora'], r=config['lora_r'])
        conf_copy['lora']['optimizer'] = optimizer

        #Train LoRA adapted network
        trainer = Trainer3D(model, dataloader, loss_fn, conf_copy)
        trainer.train(lora_dir=lora_dir)
        
        #Print performance
        model.eval()
        if config['custom_shape']:
            evaluate_custom_estimation(model, trainer)
        else:
            evaluate_sphere_estimation(model, trainer, conf_copy, config['mesh_param']['r'])
        del reference_model, model, trainer, dataloader, conf_copy

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--J', type=int, default=6, help='Number of LoRA adapted directions')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/3D/scattering/', help='Save file for weights')
    parser.add_argument('--lora_dir', type=str, default='checkpoints/3D/lora/', help='Directory containing LoRA weights')
    parser.add_argument('--hsave_dir', type=str, default = 'checkpoints/3D/phisk/', help='Save directory for PHISK weights')
    parser.add_argument('--mesh_path', type=str, default=None, help='File containing a custom shape, if None, train on a regular circle defined by mesh_param')
    args = parser.parse_args()
    
    model_name = args.model
    save_dir = args.save_dir
    lora_dir = args.save_dir
    hsave_dir = args.hsave_dir
    J = args.J
    mesh_path = args.mesh_path
    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()

    #Load config
    config_path = f"config/3D/scattering/config_{model_name}.yaml"
    config, DTYPE = load_config(config_path)
    config['preload'] = False

    with open(f"config/3D/scattering/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)

    #Load mesh
    if mesh_path is None:
        config['custom_shape'] = False
        mesh_path = generate_sphere(radius=config['mesh_param']['r'], subdivisions=8, center = config['mesh_param']['center'])
        mesh =  trimesh.load(mesh_path)
        boundary , normals = get_sphere_param(mesh.vertices)
    else :
        config['custom_shape'] = True
        mesh =  trimesh.load(mesh_path)
        normals = mesh.vertex_normals
    print(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    
    boundary = torch.tensor(boundary, dtype=DTYPE, device=config['device'])
    normals = torch.tensor(normals, dtype=DTYPE, device=config['device'])

    print('Train a reference model from scratch')
    reference_model = init_model_from_conf(config).to(config['device'])

    #Create training points
    create_dataloader3D(
    mesh=mesh,
    cache_dir='./data_points',
    config=config,
    pml_boundary=config['L']+config['pml_size'],
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
        'R': config['mesh_param']['r']
    }

    trainer = Trainer3D(reference_model, dataloader, loss_fn, config)
    trainer.train(save_dir=save_dir)


    #LoRA PINN framework
    dataloader_cpu = {
        'lora' : points['fine'],    #N_points[fine] > N_points[lora] so we ccan reuse the same one, you can regenerate points if you want
        'normals' : normals.cpu(),
        'boundary' : boundary.cpu(),
        'polygon' : mesh.vertices,
        'mesh_mask': create_3d_mesh_mask(config, mesh),
        'R': config['mesh_param']['r']
    }

    #Creating direction to adapt to
    directions = fibonacci_sphere(J, device='cpu', dtype=DTYPE)  
    directions = directions.unsqueeze(-1) 

    #Training of LoRA adapted networks
    config['lora_train'] = True
    config['lora_r'] = hconfig['r']
    print(f'Now adapting it to {J} directions with LoRA')
    multiprocess = False # SET TO TRUE IF MULTIPROCESS LEARNING IS POSSIBLE
    if multiprocess : 
        processes = []

        for i, direction in enumerate(directions):
            p = mp.Process(
                target=train_direction,
                args=(i, direction, config, model_name, dataloader_cpu, loss_fn, save_dir, lora_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else :
        for i, direction in enumerate(directions):
            train_direction(
                i, direction, config, model_name, dataloader_cpu, loss_fn, save_dir, lora_dir)
            
            gc.collect()               # clear unused Python objects
            torch.cuda.empty_cache()   # release cached CUDA memory
    print("Direction adaptation with LoRA done.")
    print("Proceed to train phisk now.")


    #PHISK Trainer initialization
    hconfig['load'] = False
    trainer = initialize_phisk_trainer3D(
        base_network=reference_model,
        hypernetwork_path=None,  # We're not using an old PHISK
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
        hconfig=hconfig,
        lora_dir=lora_dir
    )

    #PHISK training points creation
    config['adam'] = hconfig['adam']
    config['fine'] = hconfig['fine']   #IF NECESSARY AS L BFGS phase is not baseline for PHISK.

    create_dataloader3D(
    mesh=mesh,
    cache_dir='./data_points',
    config=config,
    pml_boundary=config['L']+config['pml_size'],
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
        'R': config['mesh_param']['r']
    }


    print("Training PHISK")
    trainer.train(save_dir = hsave_dir)

if __name__ == '__main__':
    main()

   