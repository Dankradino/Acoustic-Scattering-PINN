import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import init_model_from_conf
from torch.autograd import grad
from utils import green, generate_grid
from shape import generate_sphere, get_sphere_param
from Trainer.phisk3D import initialize_enhanced_trainer  # Import the corrected version
from Dataloader import create_dataloader3D, loader3D
from utils import create_3d_mesh_mask
from eval import check_wave_equation, evaluate_green_estimation, evaluate_energy_spectrum, evaluate_sphere_estimation
import yaml
import time
import trimesh
from visuals import (
    test_and_visualize_directions, 
    plot_solution_for_direction, 
    plot_attention_heatmap,
    comprehensive_direction_analysis,
    quick_test,
    run_analysis
)
from matplotlib.patches import Circle
from Trainer.phisk3D import load_directional_model

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained model')
    parser.add_argument('--res', type=int, default=100, help='Resolution of the grid')
    parser.add_argument('--L', type=float, default=5.0, help='Grid size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lora_dir', type=str, default='checkpoints/lora/', help='Directory containing LoRA weights')
    parser.add_argument('--train', type=str, default=True, help='Want to train')
    args = parser.parse_args()
    
    model_name = args.model
    preload = args.preload
    train = args.train
    lora_dir = 'checkpoints/lora3D/'
    
    torch.manual_seed(30)

    with open(f"config/scat3D/config_{model_name}.yaml") as file:
        config = yaml.safe_load(file)

    with open(f"config/scat3D/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)


    #config['fine']['batch_size'] = 32768 // 4
    # Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device  # Make sure device is in config
    config['scale'] = 1
    hconfig['adam']['batch_size'] = 4096
    hconfig['fine']['batch_size'] = 4096
    # Frequency
    f = 1000/343
    config['frequency'] = f
    
    # Create 2D grid of x points
    L = config['L']
    Lpml = config['pml_size']
    fake_bd = L + Lpml

    direction = torch.tensor([1., 1., 1.], device=device).unsqueeze(1)
    direction = direction / torch.linalg.norm(direction)
    config['direction'] = direction
    
    mesh_param = {
        'length': 1.0,
        'n_elem': 128,
        'R': 0.2,
        'cx': 0,
        'cy': 0,
        'epsilon': 1e-2,
        'num_points': 5,
        'inner_radius': 0.2,
        'outer_radius': 0.5,
        'center': (0., 0.),
        'rotation': 0.,
    }
    config['R'] = 0.2
    DTYPE = torch.float
    
    # SIREN model
    reference_model = init_model_from_conf(config).to(device)
    print('Preloading weights for reference model')
    reference_model.load_state_dict(torch.load(f'checkpoints/scattering3D/{model_name}.pth'))
    print('Done!')

    config['preload'] = preload
    loss_fn = nn.MSELoss()

    # Generate Circle Polygon
    mesh_path = generate_sphere(radius=mesh_param['R'], subdivisions=8)
    mesh =  trimesh.load(mesh_path)

    boundary , normals = get_sphere_param(mesh.vertices)
    boundary = torch.tensor(boundary, dtype=DTYPE, device=device)
    normals = torch.tensor(normals, dtype=DTYPE, device=device)

    hconfig['adam']['epochs'] = 100000
    config['adam'] = hconfig['adam']
    config['fine'] = hconfig['fine']

    create_dataloader3D(
    mesh=mesh,
    cache_dir='./data_points',
    config=config,
    pml_boundary=fake_bd,
    method='spherical_shell'
    )
    points = loader3D(config, cache_dir = './data_points')

    if preload : 
        hconfig['adam']['batch_size'] = 1
        hconfig['fine']['batch_size'] = 8192 * 2
        hconfig['fine']['batch_boundary'] = 8192 * 2 

    dataloader = {
        'adam' : points['adam'],
        'fine' : points['fine'],
        'normals' : normals,
        'boundary' : boundary,
        'polygon' : mesh.vertices,
        'mesh_mask': create_3d_mesh_mask(config, mesh),
        'R' : mesh_param['R'],

    }

    # Check if LoRA directory exists and has files
    import os
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

    hconfig['load'] = True
    # Initialize the enhanced trainer
    trainer = initialize_enhanced_trainer(
        base_network=reference_model,
        hypernetwork_path= f'checkpoints/hyper3D/{model_name}_pre_9900.pth', #None,  # We're not using the old hypernetwork
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
        hconfig=hconfig,
        lora_dir=lora_dir
    )

    if preload : 
        trainer.load_hypernetwork_checkpoint(f'checkpoints/enhanced_trainer_{model_name}_pre.pth')
    print("Enhanced trainer initialized successfully!")
    
    # Train the continuous direction control system
    print(train)
    if train:
        print('SFSDFGSERSGSFSERFSFSGFSEF')
        print("Training continuous direction control...")

        trainer.train()
            

    
    print(f"Enhanced trainer components saved to checkpoints/enhanced_trainer_{model_name}.pth")

    model = load_directional_model(
    base_network=reference_model,
    lora_dir=lora_dir, 
    checkpoint_path=f'checkpoints/enhanced_trainer_{model_name}.pth',
    config=config,
    hconfig=hconfig
    )   
    #evaluate_sphere_estimation_over_direction(model, config, mesh_param['R'])

if __name__ == '__main__':
    main()

   