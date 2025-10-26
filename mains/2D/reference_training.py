import argparse
import torch
import torch.nn as nn
import numpy as np
from model import init_model_from_conf
from shape import generate_star, generate_square, generate_circle, densify_polygon_with_normals, generate_ellipse
from Trainer import Trainer2D
from Dataloader import create_dataloader
from eval import evaluate_circle_estimation
import yaml

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained model')
    parser.add_argument('--no_train', type=bool, default = False, help='True if train mode')
    args = parser.parse_args()
    model_name = args.model
    train = not args.no_train
    preload = args.preload
    print('Train', train)
    
    torch.manual_seed(30)

    with open(f"config/scattering/config2D_{model_name}.yaml") as file:
        config = yaml.safe_load(file)


    #Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Frequency
    f = 2000/343 #25.0/(2*np.pi)
    config['frequency'] = f
    #config['device'] = device
    # Create 2D grid of x points
    L = config['L']
    Lpml = config['pml_size']
    fake_bd = L + Lpml #Set to 1

    direction =  torch.tensor([1., 1.], device = device).unsqueeze(1)     #torch.tensor([1., 1.],device = device).unsqueeze(1)
    direction =  direction / torch.linalg.norm(direction)
    config['direction'] = direction
    mesh_param = {
        'length' : 1.,
        'n_elem' : 128,
        'R' : 0.2,
        'cx' : 0,
        'cy' : 0,
        'epsilon' : 1e-2,
        'num_points' : 5,
        'inner_radius' : 0.2,
        'outer_radius' : 0.3,
        'center' : (0. ,0.),
        'rotation' : 0.,
    }

    DTYPE = torch.float
    config['R'] = mesh_param['R']
    config['center'] = torch.tensor(mesh_param['center'], dtype=DTYPE, device = device)

    #config['adam']['batch_size'] = 2*config['adam']['batch_size'] 
    #config['fine']['batch_size'] = 2*config['fine']['batch_size'] 
    # SIREN model
    siren = init_model_from_conf(config).to(device)

    if preload:
        print('Preloading weights')
        config['adam']['batch_size'] = 1
        siren.load_state_dict(torch.load(f'checkpoints/scattering/{model_name}_pre.pth'))
        print('Done !')

    config['preload'] = preload
    loss_fn = nn.MSELoss()


    #--- Generate Star Polygon ---
    
    config['shape'] = config.get('shape', 'circle')

    if config['shape']== 'circle':
        polygon = generate_circle(radius=mesh_param['R'], center=mesh_param['center'], num_points=2000)
    elif config['shape']== 'ellipse':
        polygon = generate_ellipse(rx=0.25, ry=0.75, center=(0.0, 0.0), num_points=2000, rotation=0.0)
    elif config['shape']== 'star':
        polygon = generate_star(num_points=mesh_param['num_points'], inner_radius=mesh_param['inner_radius'], outer_radius=mesh_param['outer_radius'], center=mesh_param['center'], rotation = np.pi/5)
    elif config['shape']== 'square':
        polygon = generate_square(side=mesh_param['R'], center=mesh_param['center'])
    else:
        raise ValueError(f"Unknown shape in config.")

    boundary,normals = densify_polygon_with_normals(polygon, total_points=20000)
    boundary = torch.tensor(boundary, dtype=DTYPE, device=device)
    normals = torch.tensor(normals, dtype=DTYPE, device=device)
    # circles_info = [
    # {'center': [0.0, 0.3], 'radius': 0.05},
    # {'center': [0.0, 0.1], 'radius': 0.05},
    # {'center': [0.0, -0.1], 'radius': 0.05},
    # {'center': [0.0, -0.3], 'radius': 0.05},
    # ]
#     axes = [-0.3, -0.1, 0.1, 0.3]
#     radius = 0.05
#     import itertools
#     circles_info = [
#         {'center': [x, y], 'radius': radius}
#         for x, y in itertools.product(axes, axes)
#     ]

#     def circles_info_to_polygons(circles_info, num_points=500):
#         """
#         Convert circles_info format back to polygon points if needed
#         """
#         polygons = []
#         for circle in circles_info:
#             center = circle['center']
#             radius = circle['radius']
#             polygon = generate_circle(radius=radius, center=center, num_points=num_points)
#             polygons.append(polygon)
#         return polygons
# #     polygon = [
# #     generate_circle(radius=0.05, center=(0., 0.3), num_points=500),
# #     generate_circle(radius=0.05, center=(0., 0.1), num_points=500),
# #     generate_circle(radius=0.05, center=(0., -0.1), num_points=500),
# #     generate_circle(radius=0.05, center=(0., -0.3), num_points=500),
# # ] 
#     polygon = circles_info_to_polygons(circles_info, num_points=500)

#     # Densify all
#     boundary_all = []
#     normals_all = []
#     for poly in polygon:
#         b, n = densify_polygon_with_normals(poly, total_points=5000)
#         boundary_all.append(b)
#         normals_all.append(n)

#     boundary = torch.tensor(np.vstack(boundary_all), dtype=DTYPE, device=device)
#     normals = torch.tensor(np.vstack(normals_all), dtype=DTYPE, device=device)
    dataloader = {
        'adam' : create_dataloader(polygon, config['adam'], fake_bd),
        'fine' : create_dataloader(polygon, config['fine'], fake_bd),
        'normals' : normals,
        'boundary' : boundary,
        'polygon' : polygon,

    }

    trainer = Trainer2D(siren, dataloader, loss_fn, config)

    if train:
        trainer.train()
    else:
        state_dict = torch.load(f'checkpoints/scattering/{model_name}.pth')  # ✅ load checkpoint
        siren.load_state_dict(state_dict)

    siren.eval()
    evaluate_circle_estimation(siren, config, mesh_param['R'])
    # evaluate_multiple_circles_estimation(siren, config, circles_info)

if __name__ == '__main__':
    main()