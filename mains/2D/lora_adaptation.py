import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import torch
import torch.nn as nn
import numpy as np
from model import init_model_from_conf, init_with_Lora, init_with_Lora_rff, load_directional_model
from shape import generate_circle, densify_polygon_with_normals
from Trainer import Trainer2D
from Dataloader import create_dataloader
from eval import evaluate_circle_estimation
import yaml
from eval import evaluate_circle_estimation_direction

def train_direction(i, direction, config, model_name, dataloader, loss_fn, mesh_param):
        import torch
        from copy import deepcopy
        torch.manual_seed(30 + i)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        direction =  direction / torch.linalg.norm(direction)

        print('direction' , direction)

        conf_copy = deepcopy(config)
        conf_copy['direction'] = direction

        model = init_model_from_conf(conf_copy).to(device)
        model.load_state_dict(torch.load(f'checkpoints/scattering/{model_name}.pth'))
        if model_name == 'rff':
            model, optimizer = init_with_Lora_rff(model, conf_copy['lora'], r=12, alpha=1.0, custom_activation=model[1])

        else:  
            model, optimizer = init_with_Lora(model, conf_copy['lora'], r=12)
        conf_copy['lora']['optimizer'] = optimizer

        trainer = Trainer2D(model, dataloader, loss_fn, conf_copy)
        trainer.train()

        model.eval()
        evaluate_circle_estimation(model, conf_copy, mesh_param['R'], display = False)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--retrain', type=bool, default=False, help='True if no pretrained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    model_name = args.model
    retrain = args.retrain
    
    torch.manual_seed(30)

    with open(f"config/scattering/config2D_{model_name}.yaml") as file:
        config = yaml.safe_load(file)

    config['preload'] = False

    #Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float
    # Frequency
    f = 1000/343
    config['frequency'] = f
    # Create 2D grid of x points
    L = config['L']
    Lpml = config['pml_size']
    fake_bd = L + Lpml #Set to 1

    direction = torch.tensor([1., 1.],device = device).unsqueeze(1)
    direction =  direction / torch.linalg.norm(direction)
    config['direction'] = direction
    mesh_param = {
        'length' : .5,
        'n_elem' : 128,
        'R' : 0.2,
        'cx' : 0,
        'cy' : 0,
        'epsilon' : 1e-2,
        'num_points' : 5,
        'inner_radius' : 0.2,
        'outer_radius' : 0.5,
        'center' : (0.,0.),
        'rotation' : 0.,
    }

    config['R'] = mesh_param['R']
    config['center'] = torch.tensor(mesh_param['center'], dtype = DTYPE, device = device)

    loss_fn = nn.MSELoss()
    circle_polygon = generate_circle(radius=mesh_param['R'], center=mesh_param['center'], num_points=5000)
    polygon = circle_polygon #circle_polygon  # Choose the polygon to classify against

    boundary,normals = densify_polygon_with_normals(polygon, total_points=20000)
    boundary = torch.tensor(boundary, dtype = DTYPE, device=device)
    normals = torch.tensor(normals, dtype = DTYPE, device=device)
    if retrain : 
        print('Train a model from scratch')
        model = init_model_from_conf(config).to(device)
        dataloader = {
            'adam' : create_dataloader(polygon, config['adam'], fake_bd),
            'fine' : create_dataloader(polygon, config['fine'], fake_bd),
            'lora' : create_dataloader(polygon, config['lora'], fake_bd),
            'normals' : normals,
            'boundary' : boundary,
            'polygon' : polygon,

        }
        trainer = Trainer2D(model, dataloader, loss_fn, config)
        trainer.train()
    else : 
        #config['lora']['batch_size'] = 1
        dataloader = {
            #'fine' : create_dataloader(polygon, config['fine'], fake_bd),
            'lora' : create_dataloader(polygon, config['lora'], fake_bd),
            'normals' : normals,
            'boundary' : boundary,
            'polygon' : polygon,

        }
        model = init_model_from_conf(config).to(device)
        model.load_state_dict(torch.load(f'checkpoints/scattering/{model_name}.pth'))
    print('Reference model performance :')
    evaluate_circle_estimation(model, config, mesh_param['R'], display = False)

    config['lora_train'] = True
    lora_train = False
    if lora_train :
        n_networks = 8
        theta = np.linspace(0, 2 * np.pi, n_networks, endpoint=False)
        directions = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_networks, 2)
        directions = torch.tensor(directions, dtype=DTYPE, device=device).unsqueeze(-1) # (n_networks, 2, 1)

        processes = []
        multiprocess = False
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
            

    
    with open(f"config/scattering/hconfig.yaml") as file:
        hconfig = yaml.safe_load(file)
    lora_dir = 'checkpoints/lora/'
    # SIREN model
    reference_model = init_model_from_conf(config).to(device)
    print('Preloading weights for reference model')
    reference_model.load_state_dict(torch.load(f'checkpoints/scattering/{model_name}.pth'))
    print('Done!')

    #hconfig['adam']['batch_size'] = 10
    hconfig['load'] = False    #True only if you want to load an already trained PHISK model
    config['preload'] = False
    dataloader = {
            'adam': create_dataloader(polygon, hconfig['adam'], fake_bd),
            'fine': create_dataloader(polygon, hconfig['fine'], fake_bd),
            'normals': normals,
            'boundary': boundary,
            'polygon': polygon,
        }
    from Trainer.phisk2D import initialize_phisk_trainer2D
    trainer = initialize_phisk_trainer2D(
        base_network=reference_model,
        hypernetwork_path=None, #f'checkpoints/hypernetwork/enhanced_trainer_{model_name}_pre_25000.pth', #f'checkpoints/hypernetwork/enhanced_trainer_{model_name}_pre_8000.pth', #None,  # We're not using the old hypernetwork
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
        hconfig=hconfig,
        lora_dir=lora_dir
    )

    print("Training continuous direction control...")

    trainer.train()

    print(f"Enhanced trainer components saved to checkpoints/enhanced_trainer_{model_name}.pth")
    model = load_directional_model(
    base_network=reference_model,
    lora_dir=lora_dir, 
    checkpoint_path=f'checkpoints/hypernetwork/enhanced_trainer_{model_name}_pre.pth',
    config=config,
    hconfig=hconfig
)
    #evaluate_circle_estimation_over_direction(model, config, mesh_param['R'])
    evaluate_circle_estimation_direction(model, config, mesh_param['R'])
if __name__ == '__main__':
    main()