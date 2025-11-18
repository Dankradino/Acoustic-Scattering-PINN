import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import load_config, create_2d_shape_mask
from model import init_model_from_conf, load_directional_model
from shape import generate_square, generate_star, generate_ellipse, generate_circle, densify_polygon_with_normals
from Trainer.phisk2D import initialize_phisk_trainer2D
from Dataloader import create_dataloader
from eval import evaluate_circle_estimation_direction
import yaml
from matplotlib.patches import Circle
import os

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a PHISK model (LoRA adaptation beforehand required)')
    parser.add_argument('--model', type=str, required=True, help='The model name to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained phisk model')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/2D/scattering/', help='Save directory for reference weights')
    parser.add_argument('--hsave_dir', type=str, default = 'checkpoints/2D/phisk/', help='Save directory for PHISK weights')
    parser.add_argument('--lora_dir', type=str, default='checkpoints/2D/lora/', help='Directory containing LoRA weights')
    args = parser.parse_args()
    model_name = args.model
    preload = args.preload
    save_dir = args.save_dir
    hsave_dir = args.hsave_dir
    lora_dir = args.save_dir
    
    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()
    #Load config
    config_path = f"config/2D/scattering/config.yaml"
    config, DTYPE = load_config(config_path)
    config['model'] = model_name

    with open(f"config/scattering/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)
    
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

    # Reference model
    config['preload'] = False
    reference_model = init_model_from_conf(config).to(config['device'])
    print('Preloading weights for reference model')
    reference_model.load_state_dict(torch.load(f'{save_dir}{model_name}.pth'))
    print('Done!')


    dataloader = {
            'adam': create_dataloader(polygon, hconfig['adam'], config['L']+config['pml_size']),
            'fine': create_dataloader(polygon, hconfig['fine'], config['L']+config['pml_size']),
            'normals': normals,
            'boundary': boundary,
            'polygon': polygon,
            'shape_mask' : create_2d_shape_mask(config, boundary_points = boundary),
        }

    # Check if LoRA directory exists and has files
    if not os.path.exists(lora_dir):
        print(f"Warning: LoRA directory {lora_dir} does not exist!")
        print("Please make sure you have trained LoRA weights for different directions.")
        return
    
    lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.pth')]
    if not lora_files:
        print(f"Warning: No .pth files found in {lora_dir}")
        print("Please make sure you have trained LoRA weights for different directions.")
        return
    
    print(f"Found {len(lora_files)} LoRA files: {lora_files}")


    # Initialize the PHISK trainer
    if preload: 
        hconfig['load'] = True #True if we load an old PHISK.
        trainer = initialize_phisk_trainer2D(
            base_network=reference_model,
            hypernetwork=hsave_dir,  # We're using a trained (at least partially) PHISK.
            dataloader=dataloader,
            loss_fn=loss_fn,
            config=config,
            hconfig=hconfig,
            lora_dir=lora_dir
        )
    else : 
        hconfig['load'] = False
        trainer = initialize_phisk_trainer2D(
            base_network=reference_model,
            hypernetwork=None,  # We're not using an old PHISK.
            dataloader=dataloader,
            loss_fn=loss_fn,
            config=config,
            hconfig=hconfig,
            lora_dir=lora_dir
        )

    print("PHISK trainer initialized successfully!")
    print("Training phisk")
    trainer.train(save_dir = hsave_dir)

    #Loading PHISK
    phisk = load_directional_model(
    base_network=reference_model,
    lora_dir=lora_dir, 
    checkpoint_path=f'{hsave_dir}{model_name}.pth',
    config=config,
    hconfig=hconfig
)
    if config['shape']== 'circle':
        evaluate_circle_estimation_direction(phisk, config, mesh_param['r'])

if __name__ == '__main__':
    main()

   