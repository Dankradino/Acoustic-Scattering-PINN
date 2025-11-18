import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import load_config, create_2d_shape_mask
from model import init_model_from_conf
from shape import generate_star, generate_square, generate_circle, generate_ellipse, densify_polygon_with_normals
from Trainer import Trainer2D
from Dataloader import create_dataloader
from eval import evaluate_circle_estimation

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a 2D reference model')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained model')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/2D/scattering/', help='Save file for reference weights')
    args = parser.parse_args()
    model_name = args.model
    preload = args.preload
    save_dir = args.save_dir
    
    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()

    #Load config
    config_path = f"config/2D/scattering/config.yaml"
    config, DTYPE = load_config(config_path)


    # Scatterer shape definition
    #--- Example shape you can use ---
    #Feel free to replace with a wanted shape, you only need boundary points and normal
    mesh_param = config['mesh_param']
    config['shape'] = config.get('shape', 'circle')

    if config['shape']== 'circle':
        polygon = generate_circle(radius=mesh_param['r'], center=mesh_param['center'], num_points=2000)
    elif config['shape']== 'ellipse':
        polygon = generate_ellipse(rx=mesh_param['ax'], ry=mesh_param['ay'], center=mesh_param['center'], num_points=2000, rotation=mesh_param['rotation'])
    elif config['shape']== 'star':
        polygon = generate_star(num_points=mesh_param['num_points'], inner_radius=mesh_param['inner_radius'], outer_radius=mesh_param['outer_radius'], center=mesh_param['center'], rotation = mesh_param['rotation'])
    elif config['shape']== 'square':
        polygon = generate_square(side=mesh_param['length'], center=mesh_param['center'])
    else:
        raise ValueError(f"Unknown shape in config.")

    boundary,normals = densify_polygon_with_normals(polygon, total_points=20000)
    boundary = torch.tensor(boundary, dtype=DTYPE, device=config['device'])
    normals = torch.tensor(normals, dtype=DTYPE, device=config['device'])

    # Reference model initialisation
    reference_model = init_model_from_conf(config).to(config['device'])

    config['preload'] = preload
    if preload:
        print('Preloading weights')
        config['adam']['batch_size'] = 2
        reference_model.load_state_dict(torch.load(f'{save_dir}{model_name}_pre.pth'))
        print('Done !')

    #Training data points creation
    dataloader = {
        'adam' : create_dataloader(polygon, config['adam'], config['L']+config['pml_size']),
        'fine' : create_dataloader(polygon, config['fine'], config['L']+config['pml_size']),
        'normals' : normals,
        'boundary' : boundary,
        'polygon' : polygon,
        'shape_mask' : create_2d_shape_mask(config, boundary_points = boundary),

    }

    trainer = Trainer2D(reference_model, dataloader, loss_fn, config)
    trainer.train(save_dir = save_dir)

    #Evaluation, you can replace with custom evaluation.
    reference_model.eval()
    if config['shape']=='circle':
        print('Evaluating model performance:')
        evaluate_circle_estimation(reference_model, config, mesh_param['r'])


if __name__ == '__main__':
    main()