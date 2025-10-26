import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import torch
import torch.nn as nn
import numpy as np
from Trainer.phisk2D import initialize_phisk_trainer2D
from utils import load_config, create_2d_shape_mask
from model import init_model_from_conf, init_with_Lora, init_with_Lora_rff, load_directional_model
from shape import generate_star, generate_square, generate_circle, generate_ellipse, densify_polygon_with_normals
from Trainer import Trainer2D
from Dataloader import create_dataloader
from eval import evaluate_circle_estimation
import yaml
from eval import evaluate_circle_estimation_direction

def train_direction(i, direction, config, model_name, dataloader, loss_fn, mesh_param, save_dir, lora_dir):
        '''
        Adapt a reference model to a new direction using LoRA scheme and save produced weights in lora_dir
        '''
        import torch
        from copy import deepcopy
        torch.manual_seed(30 + i)

        direction =  direction / torch.linalg.norm(direction)
        print('Adapted direction' , direction)

        conf_copy = deepcopy(config)
        conf_copy['direction'] = direction

        #Load reference model
        reference_model = init_model_from_conf(conf_copy).to(config['device'])
        reference_model.load_state_dict(torch.load(f'{save_dir}{model_name}.pth'))

        #Initialize adapted network    
        if model_name == 'rff':
            model, optimizer = init_with_Lora_rff(reference_model, conf_copy['lora'], r=12, alpha=1.0, custom_activation=model[1])
        else:  
            model, optimizer = init_with_Lora(reference_model, conf_copy['lora'], r=12)
        conf_copy['lora']['optimizer'] = optimizer

        #Train LoRA adapted network
        trainer = Trainer2D(model, dataloader, loss_fn, conf_copy)
        trainer.train(lora_dir=lora_dir)

        #Print performance
        model.eval()
        if config['shape']== 'circle':
            print(f'LoRA model performance : {direction}')
            evaluate_circle_estimation(model, conf_copy, mesh_param['R'], display = False)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a 2D reference model (optional), adapt it to J directions')
    parser.add_argument('--model', type=str, required=True, help='The model nameto use (e.g., "xxx")')
    parser.add_argument('--J', type=int, default=4, help='Number of LoRA adapted directions')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/2D/scattering/', help='Save directory for reference weights')
    parser.add_argument('--lora_dir', type=str, default = 'checkpoints/2D/lora/', help='Save directory for LoRA weights')
    parser.add_argument('--hsave_dir', type=str, default = 'checkpoints/2D/phisk/', help='Save directory for PHISK weights')
    args = parser.parse_args()
    model_name = args.model
    J = args.J
    lora_dir = args.lora_dir
    save_dir = args.save_dir
    hsave_dir = args.hsave_dir
    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()

    #Load config
    config_path = f"config/2D/scattering/config.yaml"
    config, DTYPE = load_config(config_path)

    with open(f"config/2D/scattering/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)

    # Scatterer shape definition
    #--- Example shape you can use ---
    #Feel free to replace with a wanted shape, you only need boundary points and normal
    mesh_param = config['mesh_param']
    config['shape'] = config.get('shape', 'circle')

    if config['shape']== 'circle':
        polygon = generate_circle(radius=mesh_param['r'], center=mesh_param['center'], num_points=2000)
    elif config['shape']== 'ellipse':
        polygon = generate_ellipse(rx=0.25, ry=0.75, center=(0.0, 0.0), num_points=2000, rotation=0.0)
    elif config['shape']== 'star':
        polygon = generate_star(num_points=mesh_param['num_points'], inner_radius=mesh_param['inner_radius'], outer_radius=mesh_param['outer_radius'], center=mesh_param['center'], rotation = mesh_param['rotation'])
    elif config['shape']== 'square':
        polygon = generate_square(side=mesh_param['length'], center=mesh_param['center'])
    else:
        raise ValueError(f"Unknown shape in config.")

    boundary,normals = densify_polygon_with_normals(polygon, total_points=20000)
    boundary = torch.tensor(boundary, dtype=DTYPE, device=config['device'])
    normals = torch.tensor(normals, dtype=DTYPE, device=config['device'])

    #Training of reference model
    config['preload'] = False
    print('Train a reference model')
    reference_model = init_model_from_conf(config).to(config['device'])
    dataloader = {
        'adam' : create_dataloader(polygon, config['adam'], config['L']+config['pml_size']),
        'fine' : create_dataloader(polygon, config['fine'], config['L']+config['pml_size']),
        'lora' : create_dataloader(polygon, config['lora'], config['L']+config['pml_size']),
        'normals' : normals,
        'boundary' : boundary,
        'polygon' : polygon,
        'shape_mask' : create_2d_shape_mask(config, boundary_points = boundary),

    }
    trainer = Trainer2D(reference_model, dataloader, loss_fn, config)
    trainer.train(save_dir = save_dir)
    
    #Print performance of reference model
    if config['shape']== 'circle':
        print('Reference model performance :')
        evaluate_circle_estimation(reference_model, config, mesh_param['r'], display = False)

    print(f'Now adapting it to {J} directions with LoRA')

    #Create direction to adapt to
    theta = np.linspace(0, 2 * np.pi, J, endpoint=False)
    directions = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (J, 2)
    directions = torch.tensor(directions, dtype=DTYPE, device=config['device']).unsqueeze(-1) # (J, 2, 1)

    #Training of LoRA adapted networks
    config['lora_train'] = True
    processes = []
    multiprocess = False   # SET TO TRUE IF MULTIPROCESS LEARNING IS POSSIBLE
    if multiprocess : 
        processes = []
        for i, direction in enumerate(directions):
            p = mp.Process(
                target=train_direction,
                args=(i, direction, config, model_name, dataloader, loss_fn, mesh_param)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else :
        for i, directioni in enumerate(directions):
            train_direction(
                i, directioni, config, model_name, dataloader, loss_fn, mesh_param)
        
    print("Direction adaptation with LoRA done.")
    print("Proceed to train phisk now.")
    dataloader = {
            'adam': create_dataloader(polygon, hconfig['adam'], config['L']+config['pml_size']),
            'fine': create_dataloader(polygon, hconfig['fine'], config['L']+config['pml_size']),
            'normals': normals,
            'boundary': boundary,
            'polygon': polygon,
        }
    
    trainer = initialize_phisk_trainer2D(
        base_network=reference_model,
        hypernetwork_path=None, # We're not using an old PHISK
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
        hconfig=hconfig,
        lora_dir=lora_dir
    )

    print("Training PHISK")

    trainer.train(save_dir = hsave_dir)

    print(f"PHISK saved to checkpoints/{hsave_dir}{model_name}.pth")
    model = load_directional_model(
    base_network=reference_model,
    lora_dir=lora_dir, 
    checkpoint_path=f'checkpoints/{hsave_dir}{model_name}.pth',
    config=config,
    hconfig=hconfig
)
    if config['shape']=='circle':
        print('Evaluating model performance:')
        evaluate_circle_estimation_direction(model, config, mesh_param['r'])

if __name__ == '__main__':
    main()