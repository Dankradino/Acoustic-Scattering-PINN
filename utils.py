import numpy as np
import torch
import yaml
from scipy.special import hankel1
from model.lora import LoRALinear
import hashlib
from torch.autograd import grad
from matplotlib.path import Path


'''
This modules contains many usefull function used throughout the study.
The methods are regrouped by interest.
'''

#######################################################################
# Geometry related method
#######################################################################



def generate_grid(L ,res ,dim, device = 'cpu'):
    """
    Uniform grid generation in dim dimension.
    Parameters : 
        L: Semi-length of the square/cube containing the grid
        res: Resolution of the grid (int)
        dim: Dimension of the problem (2 or 3) where grid coordinates are computed
    
    Returns:
        A uniform grid over 2D or 3D space of shape [res**dim, dim]"""
    if dim == 2:
        line = torch.linspace(-L, L, res, device=device)
        X,Y = torch.meshgrid(line, line, indexing='xy')
        grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    elif dim == 3:
        line = torch.linspace(-L, L, res, device=device)
        X,Y,Z = torch.meshgrid(line, line, line, indexing='xy')
        grid = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)
    else:
        raise ValueError("Dimension must be 2 or 3.")
    return grid
    
def fibonacci_sphere(n_points, device='cpu', dtype=torch.float32):
    """Generate approximately evenly distributed unit vectors on a sphere."""
    indices = torch.arange(0, n_points, dtype=dtype, device=device) + 0.5

    phi = torch.acos(1 - 2 * indices / n_points)       # polar angle
    theta = np.pi * (1 + 5**0.5) * indices             # golden angle * index

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    directions = torch.stack([x, y, z], dim=1)  # (n_points, 3)
    return directions


#######################################################################
# Usefull acoustic based function
#######################################################################


def green(x,y,f,c=1.):
    """
    Parameters:
        x: [N, dim] tensor of observation points
        y: [1, dim] or [N, 2] tensor of source points
        f: frequency (Hz)
        c: wave speed (default: 1.0)

    Returns:
        Tensor of shape [N, 2] representing real and imaginary parts of Green function .
    """
    alpha = 1/np.sqrt(4*np.pi)
    k = 2 * np.pi * f / c
    if x.shape[1]==2:
        r = torch.linalg.norm(x - y, dim=1).cpu().numpy()  # distance
        g = 0.25j * hankel1(0, k * r)  # complex-valued Green's function
        g = torch.tensor(g, dtype = torch.cfloat, device = x.device)
    else : 
        r = torch.linalg.norm(x-y,dim=1)
        g = alpha/r*torch.exp(-1j*(k*r))
    return torch.stack((g.real, g.imag),dim = 1)


def G_inc(x, y, f, c=1.0):
    k = 2 * np.pi * f / c  # wavenumber
    r = torch.linalg.norm(x-y[:None],dim=1)
    g = torch.exp(1j*k*r)
    g = torch.stack((g.real, g.imag), dim=-1)
    return g 

def plane_wave(x, y, f, c=1.0):
    """
    Parameters:
        x: [N, 2] tensor of observation points
        y: [1, 2] or [N, 2] tensor of source points
        f: frequency (Hz)
        c: wave speed (default: 1.0)

    Returns:
        Tensor of shape [N, 2] representing real and imaginary parts e^jkr .
    """
    r = torch.linalg.norm(x-y,dim = 1)
    k = 2 * np.pi * f / c  # wavenumber
    g = torch.exp(1j*k*r)
    return torch.stack((g.real, g.imag), dim=-1)



#######################################################################
# LoRA related method
#######################################################################

def save_lora_weights(model, path):
    '''
    Save only LoRA weights from a reference training.
    '''
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.r > 0:
            lora_state[name] = {
                'lora_A': module.lora_A.detach().cpu(),
                'lora_B': module.lora_B.detach().cpu(),
                'alpha': module.alpha,
            }

    torch.save(lora_state, path)

def load_lora_weights(model, path):
    '''
    Load only LoRA weights.
    '''
    lora_state = torch.load(path)

    for name, module in model.named_modules():
        if name in lora_state:
            state = lora_state[name]
            module.lora_A.data.copy_(state['lora_A'])
            module.lora_B.data.copy_(state['lora_B'])
            module.alpha = state['alpha']




#######################################################################
# Training ID generation
#######################################################################

def generate_config_id(config: dict) -> str:
    """
    Generate a unique ID string from a nested configuration dictionary.
    
    The output is a short hash of the sorted key-value pairs.
    """
    def flatten(d, parent_key=''):
        items = []
        for k, v in sorted(d.items()):
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key))
            else:
                items.append(f"{new_key}={v}")
        return items

    flat_items = flatten(config)
    joined = "_".join(flat_items)
    hash_id = hashlib.md5(joined.encode()).hexdigest()[:10]  # Short hash
    return f"cfg_{hash_id}"


#######################################################################
# Mask generation (for plotting)
#######################################################################

def create_2d_shape_mask(config, boundary_points):
    """Create 2D boolean mask from boundary points
    
    Args:
        config: dict with 'L' (domain half-size) and 'res' (resolution)
        boundary_points: numpy array of shape (N, 2) defining the boundary polygon
        
    Returns:
        shape_mask: 2D boolean array of shape (res, res)
    """
    # Create coordinate grids
    L = config['L']
    res = config['res']
    x = np.linspace(-L, L, res)
    y = np.linspace(-L, L, res)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Flatten coordinates for batch processing
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Create path from boundary points and check containment
    path = Path(boundary_points)
    inside_mask = path.contains_points(points)
    
    # Reshape back to 2D
    shape_mask = inside_mask.reshape(res, res)
    
    return shape_mask

def create_3d_mesh_mask(config, mesh):
    """Create 3D boolean mask from trimesh"""
    # Create coordinate grids
    L = config['L']
    res = config['res']
    x = np.linspace(-L, L, res)
    y = np.linspace(-L, L, res)
    z = np.linspace(-L, L, res)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
    
    # Flatten coordinates for batch processing
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Use trimesh's contains method (efficient batch operation)
    # Assuming self.mesh is your original trimesh object before tensor conversion
    inside_mask = mesh.contains(points)
    
    # Reshape back to 3D
    mesh_mask = inside_mask.reshape(res, res, res)
    return mesh_mask




#######################################################################
# Differential Operators
#######################################################################


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    gradi = grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return gradi


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]

    jac = torch.zeros(meta_batch_size, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status



#######################################################################
# Complex Operators
#######################################################################

def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out





#######################################################################
# Sampling Operator
#######################################################################

class HaltonBoundarySampler:
    def __init__(self, device='cuda'):
        self.device = device
        
    def halton_sequence(self, n, base):
        """Generate Halton sequence - deterministic low-discrepancy"""
        sequence = torch.zeros(n, device=self.device)
        for i in range(n):
            result = 0.0
            f = 1.0
            index = i + 1
            while index > 0:
                f /= base
                result += f * (index % base)
                index //= base
            sequence[i] = result
        return sequence
    
    def sample_boundary_halton(self, boundary, normals, batch_size, offset=0):
        """
        Faster blue noise sampling using Halton sequences
        
        Args:
            boundary: tensor of boundary points [N, 3]
            normals: tensor of normal vectors [N, 3]
            batch_size: number of samples
            offset: offset for sequence (use epoch for variation)
            
        Returns:
            boundary_points, normals_points: sampled points and normals
        """
        n_boundary = boundary.size(0)
        
        # Generate Halton sequence (bases 2 and 3 for good 2D distribution)
        h2 = self.halton_sequence(batch_size, 2)
        h3 = self.halton_sequence(batch_size, 3)
        
        # Apply offset for variation across epochs
        h2 = (h2 + offset * 0.618034) % 1.0  # Golden ratio offset
        
        # Combine sequences for better distribution
        indices_float = (h2 + h3 * 0.754877) % 1.0  # Another irrational for mixing
        idx_bd = (indices_float * n_boundary).long()
        idx_bd = torch.clamp(idx_bd, 0, n_boundary - 1)
        
        return boundary[idx_bd], normals[idx_bd]


def sample_with_blue_noise(boundary, normals, config, epoch):
    """
    Sample boundary points and normals according to Halton method
    Note : Other blue noise method can be used, justifying the general name 'sample_with_blue_noise
    eventhough the Halton does generate a true blue noise.
    """
    
    # Initialize sampler 
    if not hasattr(sample_with_blue_noise, 'sampler'):
        sample_with_blue_noise.sampler = HaltonBoundarySampler(device=boundary.device)
    
    # Sample using blue noise
    boundary_points, normals_points = sample_with_blue_noise.sampler.sample_boundary_halton(
        boundary, normals, config['batch_boundary'], offset=epoch
    )
    
    return boundary_points, normals_points



#######################################################################
# Load configuration file
#######################################################################

def load_config(config_path):
    DTYPE = torch.float # Hard-coded
    
    #Load config
    with open(config_path) as file:
        config = yaml.safe_load(file)

    #Hyperparameters for experiment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float

    direction = torch.tensor(config['ref_direction'], device = device).unsqueeze(1)
    direction =  direction / torch.linalg.norm(direction)
    config['direction'] = direction
    config['device'] = device

    #config['Z'] = torch.tensor([Re(Z), Im(Z)], dtype = DTYPE, device = device)
    #config['mode'] = 'source' but need to configurate config['source'] corresponding to source coordinate

    return config, DTYPE