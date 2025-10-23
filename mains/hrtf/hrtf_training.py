import argparse
import torch
import torch.nn as nn
import numpy as np
from model import init_model_from_conf
from shape import densify_polygon_with_normals, generate_sphere, get_sphere_param
from Trainer import HRTFTrainer
from eval import evaluate_hrtf
from utils import generate_config_id, create_3d_mesh_mask
from visuals import visualize_dataloader_points, PressureFieldVisualizer
from Dataloader import create_dataloader3D, loader3D, load_hrtf
from pysofaconventions.SOFAFile import SOFAFile
import yaml
import trimesh


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

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model and log to TensorBoard')
    parser.add_argument('--model', type=str, required=True, help='The model name or type to use (e.g., "xxx")')
    parser.add_argument('--preload', type=bool, default=False, help='True if pretrained model')
    parser.add_argument('--res', type=int, default=100, help='Resolution of the grid')
    parser.add_argument('--L', type=float, default=5.0, help='Grid size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--id', type = str, default=None, help='id of model to load')
    args = parser.parse_args()
    model_name = args.model
    preload = args.preload
    id = args.id
    torch.manual_seed(30)

    #Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float

    subject_id = "P0002"
    mesh_path = f"/home/tancrede/SONICOM/{subject_id}/3DSCAN/{subject_id}_wtt_prepd.stl" #_watertight.stl" #_wtt_prepd.stl"
    #f"/home/tancrede/SONICOM/{subject_id}/3DSCAN/{subject_id}_watertight.stl" 
    #sofa_path = f"/home/tancrede/SONICOM/{subject_id}/HRTF/HRTF/44kHz/{subject_id}_Windowed_44kHz.sofa"#_FreeFieldComp_44kHz.sofa"
    sofa_path = f"/home/tancrede/SONICOM/{subject_id}/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa"
    normalize_path = f"/home/tancrede/SONICOM/{subject_id}/3DSCAN/{subject_id}_normalize.stl"

    mesh = trimesh.load_mesh(mesh_path, process=False)

    print(f'Number of mesh vertices : {mesh.vertices.shape[0]}')

    # Conversion of mm to m
    mesh.vertices /= 1000.0  # now in meters
    #mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = mesh.bounding_box.extents.max()
    scale = 0.75 * 2.0 / scale # 0.5 * [-1, 1]
    print('Scale',scale)
    mesh.apply_scale(scale)  
    print('Max of mesh vertices', np.max(mesh.vertices))
    mesh.export(normalize_path)
    mesh_path = normalize_path
    mesh =  trimesh.load(mesh_path)

    sofa = load_hrtf(sofa_path, mesh.vertices, DTYPE, device)

    if id!=None:
        with open(f"checkpoints/hrtf/{id}/config.yaml") as file:
            config = yaml.unsafe_load(file)
            config['id'] = id
            config['adam']['batch_size'] = 2
    else:
        with open(f"config/hrtf/config_{model_name}.yaml") as file:
            config = yaml.safe_load(file)

    config['frequency'] = sofa['freqs'][2]
    print('Frequency of interest', config['frequency'])
    config['celerity'] = 343
    # Create 2D grid of x points
    L = config['L']
    Lpml = config['pml_size']
    fake_bd = L + Lpml #Set to 1


    direction = torch.tensor([1., 0., 0.],device = device).unsqueeze(1)
    print('direction', direction)
    direction =  direction / torch.linalg.norm(direction)
    config['direction'] = direction
    config['source'] = 1.5  * scale * direction.T  #Physical position of the source 1.5 meters in the experiment # * scale
    #print(sofa['mic_positions'].shape)
    #config['source'] = 0.98 * sofa['mic_positions'][0]
    print('Source relative distance' , torch.linalg.norm(config['source'], dim = -1))
    config['scale'] = scale
    config['p0'] = 10

    # Model
    model = init_model_from_conf(config).to(device)
    loss_fn = nn.MSELoss()

    config['preload'] = preload
    if id==None:
        config['id'] = generate_config_id(config)
        id = config['id']
        print('Generated id :', config['id'])

    if preload:
        model.load_state_dict(torch.load(f'checkpoints/hrtf/{id}/{model_name}_pre.pth'))

    boundary = mesh.vertices
    normals = mesh.vertex_normals
    boundary = torch.tensor(boundary, dtype=DTYPE, device=device)
    normals = torch.tensor(normals, dtype=DTYPE, device=device)
    #print('ear shape', sofa["ear_idx"])
    idx_ear = sofa["ear_idx"].cpu().numpy()  # removal of ears canal entrance in boundary conditions
    #idx_ear = np.array(idx_ear, dtype = np.int32)

    mask = np.ones(len(boundary), dtype=bool)
    mask[idx_ear] = False

    # Apply mask
    boundary = boundary[mask]
    normals = normals[mask]


    #config['adam']['batch_size'] = 4*config['adam']['batch_size'] 
    #config['adam']['batch_size'] = 4*config['adam']['batch_size'] 
    
    #
    #config['fine']['batch_size'] = 2 #2*config['fine']['batch_size'] 
    #config['fine']['batch_boundary'] = 2*config['fine']['batch_boundary'] 

    #32768
    # create_dataloader3D(mesh_path, config ,fake_bd)
    # points = loader3D(config)
#     create_ear_aware_dataloader3D(
#     mesh=mesh,
#     config=config,
#     pml_boundary=fake_bd,
#     L=L,
#     ear_positions={'positions' : sofa['mic_positions'].cpu().numpy()},
#     ear_sphere_radius= 0.1 * scale,  # x sphere around each ear
#     ear_percentage=20.0,     # X% of points near ears  #X / 2 in fact
#     method='spherical_shell'
# )
    create_dataloader3D(
    mesh=mesh,
    cache_dir='./data_points',
    config=config,
    pml_boundary=fake_bd,
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

    }

    trainer = HRTFTrainer(model, dataloader, loss_fn, config, sofa)

    trainer.train()


    model.eval()

    print('id of experiment:', config['id'])

    evaluate_hrtf(sofa, model, mesh_path, config)

if __name__ == '__main__':
    main()