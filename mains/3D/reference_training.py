import argparse
import torch
import torch.nn as nn
from model import init_model_from_conf
from shape import  generate_sphere, get_sphere_param
from Trainer import Trainer3D
from Dataloader import create_dataloader3D, loader3D
from utils import create_3d_mesh_mask, load_config
from eval import evaluate_sphere_estimation, evaluate_custom_estimation
import yaml
import trimesh

'''
Train a reference PINN for a custom shape if given or for a sphere define by config['mesh_param']
'''


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a 3D reference model')
    parser.add_argument('--model', type=str, required=True, help='The model name to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained model')
    parser.add_argument('--save_dir', type=str, default = 'checkpoints/3D/scattering/', help='Save file for reference weights')
    parser.add_argument('--mesh_path', type=str, default=None, help='File containing a custom shape, if None, train on a regular sphere defined by config[mesh_param]')
    args = parser.parse_args()
    model_name = args.model
    preload = args.preload
    save_dir = args.save_dir
    mesh_path = args.mesh_path

    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()

    #Load config
    config_path = f"config/3D/scattering/config_{model_name}.yaml"
    config, DTYPE = load_config(config_path)

    #Load mesh
    if mesh_path is None:
        config['custom_shape'] = False
        mesh_path = generate_sphere(radius=config['mesh_param']['r'], subdivisions=8, center = config['mesh_param']['center'])
        mesh =  trimesh.load(mesh_path)
        boundary , normals = get_sphere_param(mesh.vertices)
    else:
        config['custom_shape'] = True
        mesh =  trimesh.load(mesh_path)
        normals = mesh.vertex_normals

    print(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    
    boundary = torch.tensor(boundary, dtype=DTYPE, device=config['device'])
    normals = torch.tensor(normals, dtype=DTYPE, device=config['device'])

    # Reference model initialisation
    reference_model = init_model_from_conf(config).to(config['device'])

    config['preload'] = preload
    if preload:
        config['adam']['batch_size'] = 10  #Arbitrary number strictly greater than 1
        reference_model.load_state_dict(torch.load(f'{save_dir}{model_name}_pre.pth'))

    #Training data points creation
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
        'R' : config['mesh_param']['r'],
    }

    #Training 
    trainer = Trainer3D(reference_model, dataloader, loss_fn, config)
    trainer.train(save_dir=save_dir)

    #Evaluation
    reference_model.eval()
    if config['custom_shape']:
        evaluate_custom_estimation(reference_model, trainer)
    else:
        evaluate_sphere_estimation(reference_model, trainer, config, config['mesh_param']['r'])

if __name__ == '__main__':
    main()