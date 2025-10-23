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
from shape import generate_star, generate_square, generate_circle, densify_polygon_with_normals
from Trainer.phisk2D import initialize_enhanced_trainer  # Import the corrected version
from Dataloader import create_dataloader
from eval import check_wave_equation, evaluate_green_estimation, evaluate_energy_spectrum, evaluate_circle_estimation
import yaml
import time
from visuals import (
    test_and_visualize_directions, 
    plot_solution_for_direction, 
    plot_attention_heatmap,
    comprehensive_direction_analysis,
    quick_test,
    run_analysis
)
from matplotlib.patches import Circle
from Trainer.phisk2D import load_directional_model
from eval import evaluate_circle_estimation_over_direction

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
    lora_dir = 'checkpoints/lora/'
    
    torch.manual_seed(30)

    with open(f"config/scattering/config2D_{model_name}.yaml") as file:
        config = yaml.safe_load(file)

    with open(f"config/scattering/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)


    #config['fine']['batch_size'] = 32768 // 4
    # Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device  # Make sure device is in config

    # Frequency
    f = 10.0/(2*np.pi)
    config['frequency'] = f
    
    # Create 2D grid of x points
    L = config['L']
    Lpml = config['pml_size']
    fake_bd = L + Lpml

    direction = torch.tensor([1., 1.], device=device).unsqueeze(1)
    direction = direction / torch.linalg.norm(direction)
    config['direction'] = direction
    
    mesh_param = {
        'length': 1.0,
        'n_elem': 128,
        'R': 0.5,
        'cx': 0,
        'cy': 0,
        'epsilon': 1e-2,
        'num_points': 5,
        'inner_radius': 0.2,
        'outer_radius': 0.5,
        'center': (0., 0.),
        'rotation': 0.,
    }

    DTYPE = torch.float
    
    # SIREN model
    reference_model = init_model_from_conf(config).to(device)
    print('Preloading weights for reference model')
    reference_model.load_state_dict(torch.load(f'checkpoints/scattering/{model_name}.pth'))
    print('Done!')

    config['preload'] = preload
    loss_fn = nn.MSELoss()

    # Generate Circle Polygon
    circle_polygon = generate_circle(radius=mesh_param['R'], center=mesh_param['center'], num_points=20000)
    polygon = circle_polygon

    boundary, normals = densify_polygon_with_normals(polygon, total_points=20000)
    boundary = torch.tensor(boundary, dtype=DTYPE, device=device)
    normals = torch.tensor(normals, dtype=DTYPE, device=device)

    if preload : 
        hconfig['adam']['batch_size'] = 1
        hconfig['fine']['batch_size'] = 8192 * 2
        hconfig['fine']['batch_boundary'] = 8192 * 2 

    if train:
        dataloader = {
            'adam': create_dataloader(polygon, hconfig['adam'], fake_bd),
            'fine': create_dataloader(polygon, hconfig['fine'], fake_bd),
            'normals': normals,
            'boundary': boundary,
            'polygon': polygon,
        }
    else:
        dataloader = {
            'adam': create_dataloader(polygon, hconfig['adam'], fake_bd),
            'fine': create_dataloader(polygon, hconfig['fine'], fake_bd),
            'normals': normals,
            'boundary': boundary,
            'polygon': polygon,
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


    # Initialize the enhanced trainer
    trainer = initialize_enhanced_trainer(
        base_network=reference_model,
        hypernetwork=None,  # We're not using the old hypernetwork
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
    
    # Visualize interpolation
    print("\nVisualizing direction interpolation:")
    dir1 = torch.tensor([1.0, 0.0], device=device)
    dir2 = torch.tensor([0.0, 1.0], device=device)
    interpolation_results = trainer.visualize_direction_interpolation(dir1, dir2, num_steps=5)
        
    visualize = False
    if visualize:

        # Quick test for direction [1,0]
        print("="*50)
        print("QUICK TEST: Direction [1,0]")
        print("="*50)
        quick_test(trainer, device, direction=[1.0, 0.0], L=1.0)

        # Test multiple directions
        print("\n" + "="*50)
        print("TESTING MULTIPLE DIRECTIONS")
        print("="*50)
        test_and_visualize_directions(trainer, device, L=1.0)

        # Plot specific direction in detail
        print("\n" + "="*50)
        print("DETAILED PLOT: Direction [1,0]")
        print("="*50)
        direction = torch.tensor([1.0, 0.0], device=device)
        fig = plot_solution_for_direction(trainer, direction, device, L=1.0, res=128)
        if fig:
            plt.savefig('direction_1_0_solution.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Plot attention heatmap
        print("\n" + "="*50)
        print("ATTENTION MECHANISM HEATMAP")
        print("="*50)
        fig_heatmap = plot_attention_heatmap(trainer, device, L=1.0, n_dirs=16)
        if fig_heatmap:
            plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Comprehensive analysis (this will run everything)
        print("\n" + "="*50)
        print("COMPREHENSIVE ANALYSIS")
        print("="*50)
        run_analysis(trainer, device, L=1.0)

        # Test interpolation between specific directions
        print("\n" + "="*50)
        print("DIRECTION INTERPOLATION")
        print("="*50)
        dir1 = torch.tensor([1.0, 0.0], device=device)  # East
        dir2 = torch.tensor([0.0, 1.0], device=device)  # North

        interpolation_results = trainer.visualize_direction_interpolation(dir1, dir2, num_steps=5)

        # Plot each interpolation step
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, result in enumerate(interpolation_results):
            direction = torch.tensor(result['direction'], device=device)
            
            # Get solution for this direction
            params, attention_weights = trainer.get_continuous_direction_params(direction)
            
            # Create a small grid for visualization
            x = torch.linspace(-1, 1, 64, device=device)
            y = torch.linspace(-1, 1, 64, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
            
            with torch.no_grad():
                output = trainer.forward_with_lora_params(grid_points, params)
            
            # Plot magnitude
            if output.shape[1] == 2:
                field_magnitude = torch.sqrt(output[:, 0]**2 + output[:, 1]**2)
            else:
                field_magnitude = torch.abs(output[:, 0])
            
            field_magnitude = field_magnitude.cpu().numpy().reshape(64, 64)
            
            im = axes[i].imshow(field_magnitude, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
            axes[i].add_patch(Circle((0, 0), 0.5, fill=False, color='white', linewidth=1))
            axes[i].arrow(0, 0, direction[0].cpu().numpy() * 0.8, direction[1].cpu().numpy() * 0.8, 
                        head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
            axes[i].set_title(f'α={result["alpha"]:.2f}\ndir={result["direction"]}')
            axes[i].set_aspect('equal')

        plt.tight_layout()
        plt.savefig('direction_interpolation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training and testing completed successfully!")
            

    
    print(f"Enhanced trainer components saved to checkpoints/enhanced_trainer_{model_name}.pth")

    model = load_directional_model(
    base_network=reference_model,
    lora_dir=lora_dir, 
    checkpoint_path=f'checkpoints/enhanced_trainer_{model_name}.pth',
    config=config,
    hconfig=hconfig
)
    evaluate_circle_estimation_over_direction(model, config, mesh_param['R'])

if __name__ == '__main__':
    main()

   