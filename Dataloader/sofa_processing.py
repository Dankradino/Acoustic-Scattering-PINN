import numpy as np
import torch
from pysofaconventions.SOFAFile import SOFAFile

'''
This module load hrtf information from the HRTF sofa file such 
as ear positions, source position, measured HRTF values ...
'''


def load_hrtf(sofa_path, vertices, DTYPE, device):

    sofa_file = SOFAFile(sofa_path ,'r')
    
    mic_positions = sofa_file.getReceiverPositionValues()[:,:,0]
    source_positions = sofa_file.getVariableValue("SourcePosition")  
    #print('mic_positions : ', mic_positions) # like [0,-0.1, 0] [0., 0.1, 0]
    hrir_values = sofa_file.getDataIR() 
    fs = sofa_file.getSamplingRate()       
    M, R, N = hrir_values.shape 

    # Define custom axis using two ears positions in relative positions
    p1 = mic_positions[0]  # [x1, y1, z1]
    p2 = mic_positions[1]  # [x2, y2, z2]

    # Direction of the axis
    axis_dir = p2 - p1
    axis_dir = axis_dir / np.linalg.norm(axis_dir)  # Unit vector

    # Vectors from p1 to all mesh vertices
    v_to_p1 = vertices - p1

    # Project each vertex onto the axis to find closest point on the line
    proj_lengths = np.dot(v_to_p1, axis_dir)  # (N,)
    proj_points = p1 + np.outer(proj_lengths, axis_dir)  # (N, 3)

    # Compute distance from vertex to axis line
    distances = np.linalg.norm(vertices - proj_points, axis=1)

    # Get indices of two closest
    closest_indices = np.argsort(distances)[:2]
    closest_vertices = vertices[closest_indices]    #EAR ARE SUPPOSED TO BE ALIGNED ON Y AXIS

    print("Vertices corresponding to ears:")
    for i, v in enumerate(closest_vertices):
        print(f"Vertex {i+1}: {v}")

    #attributes = dir(sofa_file)
    #print(attributes)

    freqs = np.fft.fftfreq(N, d=1/fs)   # Shape: (F,)
    # FFT: Compute HRTF for all positions and both ears
    hrtf_values = np.fft.fft(hrir_values, axis=2)       # Shape: (M, R, F)

    # print("Mic Positions:", mic_positions.shape)
    # print("Source Positions:", source_positions.shape)
    # print("HRTF Values:", hrtf_values.shape)
    # print("Sampling Rate:", fs)
    # print("Frequency Axis:", freqs.shape)
    # print("FFT Values:", hrtf_values.shape)

    hrtf_values = np.stack([hrtf_values.real, hrtf_values.imag], axis=-1) 

    print('First frequencies available :' , freqs[ :15])

    freqs = torch.tensor(freqs, dtype = DTYPE, device = device)
    source_positions = torch.tensor(source_positions, dtype = DTYPE, requires_grad = False, device = device)
    mic_positions = torch.tensor(mic_positions, dtype = DTYPE, requires_grad=False,device = device)
    hrtf_values = torch.tensor(hrtf_values, dtype = DTYPE, requires_grad=False, device = device)
    closest_indices = torch.tensor(closest_indices, dtype = torch.long, requires_grad = False, device = device)
    sofa = {
        "ear_idx" : closest_indices,
        "sampling_rate" : sofa_file.getSamplingRate(),
        "sources_positions": torch.tensor(sofa_file.getVariableValue("SourcePosition"), dtype=DTYPE, requires_grad=False,device=device) ,       # [M, 3]
        "mic_positions":  torch.tensor(closest_vertices, requires_grad=False, dtype = DTYPE, device = device),   # [R, 3]
        "listener_positions": torch.tensor(sofa_file.getListenerPositionValues(), dtype=DTYPE, requires_grad=False,device=device) ,   # [1, 3]
        "freqs": freqs ,
        "hrtf": torch.tensor(hrtf_values, dtype=DTYPE,requires_grad=False, device=device) ,  # [M, R, F]
        "hrir": torch.tensor(hrir_values, dtype=DTYPE,requires_grad=False, device=device) , # [M, R, N]
    }
    return sofa


