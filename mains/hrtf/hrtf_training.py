import argparse
import torch
import torch.nn as nn
import numpy as np
from model import init_model_from_conf, init_with_Lora, init_with_Lora_rff
from Trainer import Trainer3D
from Trainer.phisk3D import initialize_phisk_trainer3D 
from eval import evaluate_hrtf
from utils import generate_config_id, create_3d_mesh_mask, load_config, fibonacci_sphere
from Dataloader import create_dataloader3D, loader3D, load_hrtf
from pysofaconventions.SOFAFile import SOFAFile
import yaml
import trimesh
import gc
import os

def Z_function(f, DTYPE, device):
    '''
    Compute the coefficient jwρ0 / Z0
    '''
    if f <= 700 :
        alpha = (f-100)/(700-100)
        Nms = alpha * 82 + (1 - alpha) * 50
    elif f <= 1000:
        alpha = (f-700)/(1000-700)
        Nms = alpha * 55 + (1 - alpha) * 82
    else: 
        Nms = 50
    Z0 = 10**(Nms/10)
    print(f'Bone impedance : {Z0}')
    rho0 = 1.21                           # Air density in kg/m^3
    w = 2 * np.pi * f                     # Angular frequency
    absorption = w * rho0 / Z0
    print(f'Resulting Robin coefficient {absorption}')
    return torch.tensor([0., absorption], dtype = DTYPE, device = device)


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

        del reference_model, model, trainer, dataloader, conf_copy


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--subject_id', type = str, default="P0002", help='id of the subject')
    parser.add_argument('--J', type=bool, default=6, help='True if pretrained model')
    parser.add_argument('--freq_idx', type = int, default=3, help='indice of frequency used from sofa file')
    args = parser.parse_args()
    
    model_name = args.model
    J = args.J
    freq_idx  = args.freq_idx
    subject_id = args.subject_id
    torch.manual_seed(30)

    #Training loss
    loss_fn = nn.MSELoss()

    #Load config
    config_path = f"config/3D/hrtf/config_{model_name}.yaml"
    config, DTYPE = load_config(config_path)
    config['hrtf'] = True
    config['custom_shape'] = True

    with open(f"config/3D/hrtf/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)

    # Load Head mesh and sofa file
    mesh_path = f"SONICOM/{subject_id}/3DSCAN/{subject_id}_wtt_prepd.stl"
    sofa_path = f"SONICOM/{subject_id}/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa"
    normalize_path = f"SONICOM/{subject_id}/3DSCAN/{subject_id}_normalize.stl"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    print(f'Number of mesh vertices : {mesh.vertices.shape[0]}')
   
    # Mesh preprocessing
    # Conversion of mm to m
    mesh.vertices /= 1000.0  # now in meters
    scale = mesh.bounding_box.extents.max()
    scale = 0.75 * 2.0 / scale # 0.5 * [-1, 1]
    print('Scale',scale)
    mesh.apply_scale(scale)  
    print('Max of mesh vertices', np.max(mesh.vertices))
    mesh.export(normalize_path)
    mesh_path = normalize_path
    mesh =  trimesh.load(mesh_path)
    boundary = mesh.vertices
    normals = mesh.vertex_normals
    boundary = torch.tensor(boundary, dtype=DTYPE, device=config['device'])
    normals = torch.tensor(normals, dtype=DTYPE, device=config['device'])

    #Load sofa and frequency definition
    sofa = load_hrtf(sofa_path, mesh.vertices, DTYPE, config['device'])
    config['frequency'] = sofa['freqs'][freq_idx]
    save_dir = f"checkpoints/hrtf/{subject_id}_{config['frequency']}/scattering/"
    lora_dir = f"checkpoints/hrtf/{subject_id}_{config['frequency']}/lora/"
    hsave_dir = f"checkpoints/hrtf/{subject_id}_{config['frequency']}/hyper/"
    directories = [save_dir, lora_dir, hsave_dir]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    print('Frequency of interest', config['frequency'])
    
    #Source position
    direction = config['direction']
    config['source'] = 1.5  * scale * direction.T  #Physical position of the source 1.5 meters in the experiment # * scale
    print('Source relative distance' , torch.linalg.norm(config['source'], dim = -1))
    config['scale'] = scale
    config['p0'] = 0.1
    config['mode'] = 'source'

    #Ear entrance canal removal
    idx_ear = sofa["ear_idx"].cpu().numpy()  # removal of ears canal entrance in boundary conditions
    mask = np.ones(len(boundary), dtype=bool)
    mask[idx_ear] = False
    boundary = boundary[mask]
    normals = normals[mask]

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
        reference_network=reference_model,
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