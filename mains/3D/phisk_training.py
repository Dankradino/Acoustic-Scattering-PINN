import argparse
import torch
import torch.nn as nn
from model import init_model_from_conf, load_directional_model
from shape import generate_sphere, get_sphere_param
from Trainer.phisk3D import initialize_phisk_trainer3D 
from Dataloader import create_dataloader3D, loader3D
from utils import create_3d_mesh_mask, load_config
import yaml
import trimesh
import os

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a PHISK model (LoRA adaptation beforehand required)')
    parser.add_argument('--model', type=str, required=True, help='The model name to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained phisk model')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/3D/scattering/', help='Save directory for reference weights')
    parser.add_argument('--hsave_dir', type=str, default = 'checkpoints/3D/phisk/', help='Save directory for PHISK weights')
    parser.add_argument('--lora_dir', type=str, default='checkpoints/3D/lora/', help='Directory containing LoRA weights')
    parser.add_argument('--mesh_path', type=str, default=None, help='File containing a custom shape, if None, train on a regular sphere defined by config[mesh_param]')
    args = parser.parse_args()
    
    model_name = args.model
    preload = args.preload
    save_dir = args.save_dir
    hsave_dir = args.hsave_dir
    lora_dir = args.save_dir
    mesh_path = args.mesh_path
    
    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()

    #Load config
    config_path = f"config/3D/scattering/config_{model_name}.yaml"
    config, DTYPE = load_config(config_path)
    config['model'] = model_name

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

    # Reference model loading
    reference_model = init_model_from_conf(config).to(config['device'])
    print('Preloading weights for reference model')
    reference_model.load_state_dict(torch.load(f'{save_dir}{model_name}.pth'))
    print('Done!')

    config['preload'] = False

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

    #Training data points creation
    config['adam'] = hconfig['adam']
    config['fine'] = hconfig['fine']
    create_dataloader3D(
    mesh=mesh,
    cache_dir='./data_points',
    config=config,
    pml_boundary=config['L']+config['pml_size'],
    method='spherical_shell'
    )
    points = loader3D(config, cache_dir = './data_points')

    dataloader = {
        'adam' : points['adam'],
        'fine' : points['fine'],
        'normals' : normals,
        'boundary' : boundary,
        'polygon' : mesh.vertices,
        'mesh_mask': create_3d_mesh_mask(config, mesh),
        'R' : config['mesh_param']['r']
    }
    # Initialize PHISK trainer
    if preload:
        hconfig['load'] = True  #True if we load an old PHISK.
        trainer = initialize_phisk_trainer3D(
            base_network=reference_model,
            hypernetwork_path= hsave_dir,  # We're using a trained (at least partially) PHISK.
            dataloader=dataloader,
            loss_fn=loss_fn,
            config=config,
            hconfig=hconfig,
            lora_dir=lora_dir

        )
    else:
        hconfig['load'] = False
        trainer = initialize_phisk_trainer3D(
            base_network=reference_model,
            hypernetwork_path= None,  # We're not using an old PHISK.
            dataloader=dataloader,
            loss_fn=loss_fn,
            config=config,
            hconfig=hconfig,
            lora_dir=lora_dir

        )
    print("Phisk Trainer initialized successfully!")
    
    # Train PHISK
    print("Training phisk")
    trainer.train(save_dir=hsave_dir)
    print(f"Training done and saved under {hsave_dir}{model_name}.pth")


if __name__ == '__main__':
    main()

   