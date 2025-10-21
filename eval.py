import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_grid, green
from Trainer.utils import compl_mul, compl_div
from scipy.special import hankel1,jv, spherical_jn, spherical_yn, sph_harm
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import trimesh


'''
This modules introduces every component for evaluation of the model.
'''


########################################################
# Evalutation metrics
########################################################

def nmse(prediction, ground_truth):
    """
    Compute Normalized Mean Squared Error (NMSE).
    
    Parameters:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
    
    Returns:
        float: NMSE value
    """
    mse = torch.mean((prediction - ground_truth) ** 2)
    variance = torch.mean((ground_truth - torch.mean(ground_truth)) ** 2)
    target_square = torch.mean(ground_truth**2)
    
    if variance == 0:
        return torch.tensor(float('inf')) if mse != 0 else torch.tensor(0.0)

    return 10*torch.log10(mse / target_square)

def R2_L2(prediction, ground_truth):
    """
    Compute R2 Score and L2 Score for complex field.
    
    Parameters:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
    
    Returns:
        float: NMSE value
    """
    prediction = prediction[:,0] + 1j * prediction[:,1]
    ground_truth = ground_truth[:,0] + 1j * ground_truth[:,1]
    mse = torch.mean(torch.abs(prediction - ground_truth) ** 2)
    variance = torch.mean(torch.abs(ground_truth - torch.mean(ground_truth)) ** 2)
    target_square = torch.mean(torch.abs(ground_truth)**2)
    
    if variance == 0:
        return torch.tensor(float('inf')) if mse != 0 else torch.tensor(0.0)
    print(f'MSE: {mse.item():.4e}')
    print(f'Variance: {variance.item():.4e}')
    r2_score = 1 - mse / variance
    print('R2 Score : {:.2e}'.format(r2_score))
    print('L2 Score :', torch.sqrt(mse/target_square))
    Nmse = 10*torch.log10(mse / target_square)
    print(f'NMSE complex : {Nmse.item():.4e}')
    return 

def cosine_similarity(predictions, targets):
    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    cos_sim = F.cosine_similarity(predictions, targets, dim=0)
    return 1-cos_sim


########################################################
# Spectrum visualisation
########################################################

def energy_spectrum_complex_real_imag(p):
    """
    Compute energy spectrum for complex field stored as (2, R, R) real/imag parts.
    To visualize pressure spectrum repartition.
    Args:
      p (torch.Tensor): real tensor shape (2, R, R)
      
    Returns:
      spec (torch.Tensor): radial energy spectrum (1D)
      k_bin (torch.Tensor): wave numbers
    """
    assert p.shape[0] == 2, "Input should have shape (2, R, R) for real and imag parts."
    res = p.shape[-2:]
    assert res[0] == res[1], "Spatial dimensions must be equal."
    
    # Convert to complex tensor
    p_complex = torch.complex(p[0], p[1])
    
    # FFT and shift
    p_hat = torch.fft.fft2(p_complex)
    p_hat = torch.fft.fftshift(p_hat)
    
    dims = res[0] * res[1]
    E_k = (p_hat.abs() ** 2) / dims**2  # Power spectral density
    
    box_size_x, box_size_y = E_k.shape
    centerx, centery = box_size_x // 2, box_size_y // 2
    
    box_radius = int(np.ceil(np.sqrt(box_size_x**2 + box_size_y**2) / 2) + 1)
    E_avsphr = torch.zeros(box_radius, device=p.device)
    
    # Radial averaging
    for i in range(box_size_x):
        for j in range(box_size_y):
            r = int(round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2)))
            E_avsphr[r] += E_k[i, j]
    
    k_bin = torch.arange(len(E_avsphr), device=p.device)
    return E_avsphr, k_bin



########################################################
# 2D evaluation of the scattering estimatation
########################################################


def sound_hard_circle(config, points, rad=0.5, center = np.array([0., 0.])):
    '''
    Analytical solution for the circle sound-hard scattering problem.
    '''
    k = config['frequency'] * 2 * np.pi
    a = rad
    points = points.cpu().numpy()

    fem_xx = points[:, 0]
    fem_xy = points[:, 1]

    r = np.sqrt(fem_xx**2 + fem_xy**2)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = len(fem_xx)

    # Direction
    d = config.get("direction", [1.0, 0.0])
    d = d.cpu().numpy()
    phi = np.arctan2(d[1], d[0])  # incoming angle

    # Incident wave
    u_inc = np.exp(1j * k * (fem_xx * d[0] + fem_xy * d[1]))
    u_inc[r < a] = 0.0  # inside the obstacle

    # Scattered field
    u_scat = np.zeros(npts, dtype=np.complex128)
    n_terms = int(30 + (k * a) * 1.01)

    for n in range(-n_terms, n_terms + 1):
        # Derivatives
        Jn = jv(n, k * a)
        Jn1 = jv(n - 1, k * a)
        Hn = hankel1(n, k * a)
        Hn1 = hankel1(n + 1, k * a)

        bessel_deriv = Jn1 - n / (k * a) * Jn
        hankel_deriv = n / (k * a) * Hn - Hn1

        coeff = -1j**n * (bessel_deriv / hankel_deriv)
        u_scat += coeff * hankel1(n, k * r) * np.exp(1j * n * (theta - phi))

    u_scat[r < a] = 0.0

    x0 = center[0]
    y0 = center[1]
    if not (x0 == 0 and y0 == 0):
        phase_shift = np.exp(1j * k * (d[0] * x0 + d[1] * y0))
        u_scat *= phase_shift
    #u_tot = u_scat + u_inc  # total field

    return torch.tensor(np.stack([u_scat.real, u_scat.imag], axis=1)).cuda()

def evaluate_circle_estimation(model, config, R, display = True):
    """
    Evaluate reference model performance.
    
    Args:
        model: The neural network model
        config: Configuration dictionary
        R: Circle radius
        plot: Whether to create plots (default: True)
    
    Returns:
        dict: Dictionary containing all metrics and angles
    """
    L = config['L']
    res = config['res']
    x_grid = generate_grid(L, res, 2, device=model.device)
    mask = (x_grid[:,0]-config['center'][0])**2 + (x_grid[:,1]-config['center'][1])**2 > config['R']**2
    print('mask shape :' ,mask.shape)
    k = 2 * np.pi * config['frequency'] / config['celerity']  # wavenumber
    with torch.no_grad():
        model.eval()
        prediction = torch.zeros((res**2,2), device = config['device'])
        target = torch.zeros((res**2,2), device = config['device'], dtype = torch.double)
        prediction[mask,:] = model(x_grid[mask,:])
        target[mask,:] = sound_hard_circle(config, x_grid[mask,:] - config['center'], R, config['center'].cpu().numpy())

    nmse_real = nmse(prediction[: ,0], target[: ,0])
    nmse_imag = nmse(prediction[: ,1], target[: ,1])

    cos_sim_real = cosine_similarity(prediction[:, 0], target[:, 0])
    cos_sim_imag = cosine_similarity(prediction[:, 1], target[:, 1])

    R2_L2(prediction, target)
    print(f"NMSE Real: {nmse_real.item():.4f}, "
      f"NMSE Imag: {nmse_imag.item():.4f}")
    print(f"Cosine Similarity Real: {cos_sim_real.item()}, Cosine Similarity Imag: {cos_sim_imag.item()}")

    # Errors
    if display : 
        error = torch.abs(prediction - target).cpu().numpy()
        error_real = error[:,0].reshape(res, res)
        error_imag = error[:,1].reshape(res, res)
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))

        p_re = prediction[: ,0].cpu().numpy()
        p_re = p_re.reshape(res, res)
        p_im = prediction[: ,1].cpu().numpy()
        p_im = p_im.reshape(res, res)
        im0 = axs[0, 0].imshow(p_re, extent=[-L, L, -L, L], origin='lower', cmap = 'plasma')
        axs[0, 0].set_title('Estimated Real Part')
        fig.colorbar(im0, ax=axs[0, 0])

        im1 = axs[1, 0].imshow(p_im, extent=[-L, L, -L, L], origin='lower', cmap = 'plasma')
        axs[1, 0].set_title('Estimated Imaginary Part')
        fig.colorbar(im1, ax=axs[1, 0])

        im2 = axs[0, 1].imshow(error_real, extent=[-L, L, -L, L], origin='lower', cmap = 'plasma')
        axs[0, 1].set_title('Absolute Error (Real Part)')
        fig.colorbar(im2, ax=axs[0, 1])

        im1 = axs[1, 1].imshow(error_imag, extent=[-L, L, -L, L], origin='lower', cmap = 'plasma')
        axs[1, 1].set_title('Absolute Error (Imaginary Part)')
        fig.colorbar(im1, ax=axs[1, 1])

        plt.tight_layout()
        plt.savefig("siren_error_plot.png", dpi=300, bbox_inches='tight')
        plt.savefig("siren_error_plot.eps", dpi=300, bbox_inches='tight')
        plt.show()
    return nmse_real, nmse_imag, cos_sim_real, cos_sim_imag


def evaluate_circle_estimation_direction(model, config, R, plot=True,  num_dir = 90):
    """
    Evaluate model performance across different incident wave directions
    
    Args:
        model: The neural network model
        config: Configuration dictionary
        R: Circle radius
        plot: Whether to create plots (default: True)
    
    Returns:
        dict: Dictionary containing all metrics and angles
    """
    L = config['L']
    res = config['res']
    x_grid = generate_grid(L, res, 2, device=model.device)
    mask = x_grid[:,0]**2 + x_grid[:,1]**2 > R**2
    print('mask shape:', mask.shape)
    
    k = 2 * np.pi * config['frequency'] / config['celerity']  # wavenumber
    theta_batch = torch.linspace(0, 2 * np.pi, num_dir)
    direction_batch = torch.stack((torch.cos(theta_batch), torch.sin(theta_batch)), dim=1).to(config['device'])
    
    # Initialize lists to store results
    angles = []
    nmse_real_list = []
    nmse_imag_list = []
    nmse_real_std_list = []
    nmse_imag_std_list = []
    cos_sim_real_list = []
    cos_sim_imag_list = []
    
    # Evaluate for each direction
    for i, direction in enumerate(direction_batch):
        config['direction'] = direction.T
        
        with torch.no_grad():
            model.eval()
            prediction = torch.zeros((res**2, 2), device=config['device'])
            target = torch.zeros((res**2, 2), device=config['device'], dtype=torch.double)
            prediction[mask, :] = model(x_grid[mask, :], direction)
            target[mask, :] = sound_hard_circle(config, x_grid[mask, :], R)
            
            # Compute metrics
            nmse_real = nmse(prediction[:, 0], target[:, 0])
            nmse_imag = nmse(prediction[:, 1], target[:, 1])
            cos_sim_real = cosine_similarity(prediction[:, 0], target[:, 0])
            cos_sim_imag = cosine_similarity(prediction[:, 1], target[:, 1])
            
            # Compute error statistics
            real_error = prediction[:, 0] - target[:, 0]
            imag_error = prediction[:, 1] - target[:, 1]
            
            # Per-sample NMSE values
            nmse_real_vals = (real_error ** 2) / (target[:, 0] ** 2 + 1e-8)
            nmse_imag_vals = (imag_error ** 2) / (target[:, 1] ** 2 + 1e-8)
            
            # Convert to dB and compute std
            nmse_real_dB_vals = 10 * torch.log10(nmse_real_vals + 1e-12)
            nmse_imag_dB_vals = 10 * torch.log10(nmse_imag_vals + 1e-12)
            nmse_real_std = nmse_real_dB_vals.std()
            nmse_imag_std = nmse_imag_dB_vals.std()
            
            # Store results
            angle_deg = theta_batch[i].item() * 180 / np.pi
            angles.append(angle_deg)
            nmse_real_list.append(nmse_real.item())
            nmse_imag_list.append(nmse_imag.item())
            nmse_real_std_list.append(nmse_real_std.item())
            nmse_imag_std_list.append(nmse_imag_std.item())
            cos_sim_real_list.append(cos_sim_real.item())
            cos_sim_imag_list.append(cos_sim_imag.item())
            
            # Print progress
            if i % 1 == 0 or i == len(direction_batch) - 1:
                print(f"Direction {i+1}/{len(direction_batch)}: {angle_deg:.1f}°")
                print(f"  NMSE Real: {nmse_real.item():.4f} ± {nmse_real_std.item():.4f}")
                print(f"  NMSE Imag: {nmse_imag.item():.4f} ± {nmse_imag_std.item():.4f}")
                print(f"  Cosine Similarity Real: {cos_sim_real.item():.4f}, Imag: {cos_sim_imag.item():.4f}")
    
    # Create comprehensive plots
    if plot:
        create_evaluation_plots(angles, nmse_real_list, nmse_imag_list, 
                              nmse_real_std_list, nmse_imag_std_list,
                              cos_sim_real_list, cos_sim_imag_list, R)
    
    # Return all results
    results = {
        'angles': angles,
        'nmse_real': nmse_real_list,
        'nmse_imag': nmse_imag_list,
        'nmse_real_std': nmse_real_std_list,
        'nmse_imag_std': nmse_imag_std_list,
        'cos_sim_real': cos_sim_real_list,
        'cos_sim_imag': cos_sim_imag_list,
        'radius': R
    }
    
    return results

def create_evaluation_plots(angles, nmse_real, nmse_imag,
                          cos_sim_real, cos_sim_imag, R):
    """Create comprehensive evaluation plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: NMSE in dB (your values are already in dB)
    ax1.plot(angles, nmse_real, 'o-', label='Real', linewidth=2, markersize=4)
    ax1.plot(angles, nmse_imag, 's-', label='Imaginary', linewidth=2, markersize=4)
    ax1.set_xlabel('Incident Direction (degrees)')
    ax1.set_ylabel('NMSE (dB)')
    ax1.set_title(f'NMSE (dB) vs Incident Direction (R={R})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convert to linear scale for comparison
    nmse_real_linear = 10 ** (np.array(nmse_real) / 10)
    nmse_imag_linear = 10 ** (np.array(nmse_imag) / 10)
    ax2.plot(angles, nmse_real_linear, 'o-', label='Real', linewidth=2, markersize=4)
    ax2.plot(angles, nmse_imag_linear, 's-', label='Imaginary', linewidth=2, markersize=4)
    ax2.set_xlabel('Incident Direction (degrees)')
    ax2.set_ylabel('NMSE (linear scale)')
    ax2.set_title(f'NMSE (Linear Scale) vs Incident Direction (R={R})')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Plot 3: Cosine Similarity
    ax3.plot(angles, cos_sim_real, 'o-', label='Real', linewidth=2, markersize=4)
    ax3.plot(angles, cos_sim_imag, 's-', label='Imaginary', linewidth=2, markersize=4)
    ax3.set_xlabel('Incident Direction (degrees)')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title(f'Cosine Similarity vs Incident Direction (R={R})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.95, 1.0])
    
    # Plot 4: Polar plot of NMSE
    ax4 = plt.subplot(224, projection='polar')
    angles_rad = np.array(angles) * np.pi / 180
    ax4.plot(angles_rad, nmse_real, 'o-', label='Real', linewidth=2, markersize=4)
    ax4.plot(angles_rad, nmse_imag, 's-', label='Imaginary', linewidth=2, markersize=4)
    ax4.set_title(f'NMSE (dB) - Polar View (R={R})', pad=20)
    ax4.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
    ax4.grid(True)
    
    
    plt.tight_layout()
    plt.show()
    
    # Additional summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Circle Radius: {R}")
    print(f"Number of directions tested: {len(angles)}")
    print(f"Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
    print("\nNMSE Statistics:")
    print(f"  Real - Mean: {np.mean(nmse_real):.2e}, Std: {np.std(nmse_real):.2e}")
    print(f"  Imag - Mean: {np.mean(nmse_imag):.2e}, Std: {np.std(nmse_imag):.2e}")
    print("\nCosine Similarity Statistics:")
    print(f"  Real - Mean: {np.mean(cos_sim_real):.4f}, Std: {np.std(cos_sim_real):.4f}")
    print(f"  Imag - Mean: {np.mean(cos_sim_imag):.4f}, Std: {np.std(cos_sim_imag):.4f}")
    
    # Find best and worst performing directions
    best_real_idx = np.argmin(nmse_real)
    worst_real_idx = np.argmax(nmse_real)
    best_imag_idx = np.argmin(nmse_imag)
    worst_imag_idx = np.argmax(nmse_imag)
    
    print(f"\nBest Performance:")
    print(f"  Real: {angles[best_real_idx]:.1f}° (NMSE: {nmse_real[best_real_idx]:.2e})")
    print(f"  Imag: {angles[best_imag_idx]:.1f}° (NMSE: {nmse_imag[best_imag_idx]:.2e})")
    print(f"\nWorst Performance:")
    print(f"  Real: {angles[worst_real_idx]:.1f}° (NMSE: {nmse_real[worst_real_idx]:.2e})")
    print(f"  Imag: {angles[worst_imag_idx]:.1f}° (NMSE: {nmse_imag[worst_imag_idx]:.2e})")


def evaluate_energy_spectrum(model, config, R):
    '''
    Visualize the energy spectrum difference between analytical and estimated field.
    '''
    L = config['L']
    res = config['res']
    mask = x_grid[:,0]**2 + x_grid[:,1]**2 > R**2
    x_grid = generate_grid(L, res, 2, device=model.device)
    k = 2 * np.pi * config['frequency'] / config['celerity']  # wavenumber
    with torch.no_grad():
        model.eval()
        prediction = torch.zeros((res**2,2), device = config['device'])
        target = torch.zeros((res**2,2), device = config['device'], dtype = torch.double)
        prediction[mask,:] = model(x_grid[mask,:])
        target[mask,:] = sound_hard_circle(config, x_grid[mask,:], R)

    # Compute energy spectrum
    uv = prediction.reshape(res,res,2).permute(2, 0, 1) #.transpose(2, 0, 1)
    uv_true = target.reshape(res,res,2).permute(2, 0, 1) 

    E_spec_pred, k_pred = energy_spectrum_complex_real_imag(uv)  
    E_spec_true, k_true = energy_spectrum_complex_real_imag(uv_true)  

    plt.figure(figsize=(8,6))
    plt.loglog(k_pred.cpu(), E_spec_pred.cpu(), label='Predicted')
    plt.loglog(k_true.cpu(), E_spec_true.cpu(), label='True / Ground Truth')
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy Spectrum E(k)')
    plt.title('Energy Spectrum from Complex Pressure Field')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


########################################################
# 3D evaluation of the scattering estimation for a sphere
########################################################

def evaluate_sphere_estimation(model, trainer, config, R, display = False):
    L = config['L']
    x_grid = trainer.x_grid
    target = torch.tensor(trainer.solution, device = trainer.device)
    mask = x_grid[:,0]**2 + x_grid[:,1]**2 + x_grid[:, 2]**2> R**2               
    res = config['res']
    with torch.no_grad():
        model.eval()
        prediction = torch.zeros((res**3,2), device = config['device'])
        prediction[mask,:] = model(x_grid[mask,:])
    R2_L2(prediction, target)
    nmse_real = nmse(prediction[:,0], target[:,0])
    nmse_imag = nmse(prediction[:,1], target[:,1])

    cos_sim_real = cosine_similarity(prediction[:,0], target[:,0])
    cos_sim_imag = cosine_similarity(prediction[:,1], target[:,1])

    print(f"NMSE Real: {nmse_real.item():.4f}, "
      f"NMSE Imag: {nmse_imag.item():.4f}")
    print(f"Cosine Similarity Real: {cos_sim_real.item()}, Cosine Similarity Imag: {cos_sim_imag.item()}")

    # Errors
    if display:
        error = torch.abs(prediction - target).cpu().numpy()
        error_real = error[:,0].reshape(res, res, res)[: , :, res//2]
        error_imag = error[:,1].reshape(res, res, res)[: , :, res//2]


        z_choice = res//2
        format = (slice(None), slice(None), z_choice)
        u_real = prediction[:, 0].reshape(res, res, res)
        u_imag = prediction[:, 1].reshape(res, res, res)
        u_real = u_real[format].cpu().numpy()
        u_imag = u_imag[format].cpu().numpy()
        target = np.zeros((res,res,2))
        target[...,0] = trainer.reshape_solution[format + (0,)] 
        target[...,1] = trainer.reshape_solution[format + (1,)] 
        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        # Real part plot
        im0 = axs[0, 0].imshow(u_real, extent=[-L, L, -L, L], origin='lower', cmap='viridis')
        #axs[0, 0].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0, 0].set_title('Scattered field real part')
        fig.colorbar(im0, ax=axs[0, 0])

        # Imaginary part plot
        im1 = axs[0,1].imshow(u_imag, extent=[-L, L, -L, L], origin='lower', cmap='viridis')
        #axs[0,1].add_patch(create_obstacle_patch(polygon, shape_type="polygon"))
        axs[0,1].set_title('Scattered field imag part')
        fig.colorbar(im1, ax=axs[0, 1])

        # Real part plot
        im2 = axs[1, 0].imshow(error_real , extent=[-L, L, -L, L], origin='lower', cmap='viridis')
        axs[1, 0].set_title('Absolute Error (Real Part)')
        fig.colorbar(im2, ax=axs[1, 0])

        # Imaginary part plot
        im3 = axs[1,1].imshow(error_imag , extent=[-L, L, -L, L], origin='lower', cmap='viridis')
        axs[1,1].set_title('Absolute Error (Imaginary Part)')
        fig.colorbar(im3, ax=axs[1, 1])
        plt.tight_layout()
        plt.show()



class AcousticScattering3D:
    def __init__(self, ka_max=20, n_terms=50, incident_direction=None, sphere_radius=1.0, device='cpu'):
        """
        Acoustic scattering by a sound-hard sphere with PyTorch grid support
        
        Parameters:
        ka_max: maximum size parameter for calculations
        n_terms: number of terms in the series expansion
        incident_direction: incident wave direction as (x,y,z) vector
        sphere_radius: radius of the scattering sphere
        device: PyTorch device ('cpu' or 'cuda')
        """
        self.ka_max = ka_max
        self.n_terms = n_terms
        self.sphere_radius = sphere_radius
        self.device = device
        
        # Set incident direction
        if incident_direction is None:
            self.incident_direction = np.array([0, 0, 1])  # Default: +z direction
        else:
            self.incident_direction = np.array(incident_direction).T[0]
            self.incident_direction = self.incident_direction / np.linalg.norm(self.incident_direction)
        
        # Convert to spherical coordinates
        self.incident_theta = np.arccos(np.clip(self.incident_direction[2], -1, 1))
        self.incident_phi = np.arctan2(self.incident_direction[1], self.incident_direction[0])
        
        print(f"Incident direction: {self.incident_direction}")
        #print(f"Incident theta: {self.incident_theta:.3f}, phi: {self.incident_phi:.3f}")
    
    def spherical_hankel1(self, n, x):
        """Spherical Hankel function of first kind"""
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    
    def spherical_jn_derivative(self, n, x):
        """Derivative of spherical Bessel function j_n(x)"""
        if n == 0:
            return -spherical_jn(1, x)
        else:
            return spherical_jn(n-1, x) - (n+1)/x * spherical_jn(n, x)
    
    def spherical_hankel1_derivative(self, n, x):
        """Derivative of spherical Hankel function h_n^(1)(x)"""
        if n == 0:
            return -self.spherical_hankel1(1, x)
        else:
            return self.spherical_hankel1(n-1, x) - (n+1)/x * self.spherical_hankel1(n, x)
    
    def scattering_coefficients(self, ka):
        """Calculate scattering coefficients A_n for Neumann boundary condition"""
        A_n = np.zeros(self.n_terms, dtype=complex)
        
        for n in range(self.n_terms):
            jn_prime = self.spherical_jn_derivative(n, ka)
            hn_prime = self.spherical_hankel1_derivative(n, ka)
            
            if abs(hn_prime) > 1e-15:
                A_n[n] = -jn_prime / hn_prime
            else:
                A_n[n] = 0
        
        return A_n
    
    def incident_field_pytorch(self, grid, k, amplitude=1.0):
        """
        Calculate incident plane wave field on PyTorch grid
        
        Parameters:
        grid: PyTorch tensor of shape (N, 3) with coordinates
        k: wave number
        amplitude: incident wave amplitude
        
        Returns:
        U_inc: PyTorch tensor of shape (N,) with incident field values
        """
        # Convert to numpy for calculation, then back to torch
        grid_np = grid.cpu().numpy()
        if isinstance(k, torch.Tensor):
            k = k.detach().cpu().numpy()
        # Calculate k·r for each grid point
        k_dot_r = (k * self.incident_direction[0] * grid_np[:, 0] + 
                   k * self.incident_direction[1] * grid_np[:, 1] + 
                   k * self.incident_direction[2] * grid_np[:, 2])
        
        # Incident field
        U_inc = amplitude * np.exp(1j * k_dot_r)
        
        # Convert back to torch
        U_inc_torch = torch.from_numpy(U_inc).to(self.device)
        
        return U_inc_torch
    
    def scattered_field_pytorch(self, grid, ka, amplitude=1.0):
        """
        Calculate scattered field on PyTorch grid using spherical harmonics expansion
        
        Parameters:
            grid: PyTorch tensor of shape (N, 3) with coordinates
            ka: size parameter
            amplitude: incident wave amplitude
            
        Returns:
        U_scat: PyTorch tensor of shape (N,) with scattered field values
        """
        k = ka / self.sphere_radius
        if isinstance(k, torch.Tensor):
            ka = ka.detach().cpu().numpy()
            k = k.detach().cpu().numpy()
        A_n = self.scattering_coefficients(ka)
        
        # Convert to numpy for calculation
        grid_np = grid.cpu().numpy()
        
        # Convert to spherical coordinates
        x, y, z = grid_np[:, 0], grid_np[:, 1], grid_np[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / (r + 1e-15), -1, 1))  # Avoid division by zero
        phi = np.arctan2(y, x)
        
        # Initialize scattered field
        U_scat = np.zeros(len(grid_np), dtype=complex)
        
        # Calculate scattered field using spherical harmonics expansion
        for n in range(self.n_terms):
            if abs(A_n[n]) > 1e-15:  # Only calculate non-zero terms
                # Sum over all m from -n to n
                for m in range(-n, n+1):
                    # For points outside sphere
                    outside_mask = r > self.sphere_radius
                    
                    if np.any(outside_mask):
                        # Spherical Hankel function
                        h_n = self.spherical_hankel1(n, k * r[outside_mask])
                        
                        # Spherical harmonic at observation points
                        Y_nm = sph_harm(m, n, phi[outside_mask], theta[outside_mask])
                        
                        # Incident field expansion coefficient
                        # For plane wave: exp(ik·r) = 4π * Σ i^n * j_n(kr) * Y_n^m*(k̂) * Y_n^m(r̂)
                        # We need Y_n^m*(incident direction)
                        Y_nm_inc_conj = np.conj(sph_harm(m, n, self.incident_phi, self.incident_theta))
                        
                        # Coefficient for plane wave expansion
                        coeff = 4 * np.pi * (1j)**n * Y_nm_inc_conj
                        
                        # Add contribution from this (n,m) term
                        U_scat[outside_mask] += amplitude * A_n[n] * coeff * h_n * Y_nm
        
        # Convert back to torch
        U_scat_torch = torch.from_numpy(U_scat).to(self.device)
        
        return U_scat_torch
    
    def total_field_pytorch(self, L, res, ka, amplitude=1.0):
        """
        Calculate total acoustic field U = U_inc + U_scat on your PyTorch grid
        
        Parameters:
        L: half domain size (grid goes from -L to L)
        res: resolution (number of points per dimension)
        ka: size parameter
        amplitude: incident wave amplitude
        
        Returns:
        U: PyTorch tensor of shape (res^3,) with total field values
        grid: PyTorch tensor of shape (res^3, 3) with coordinates
        """
        # Generate grid using your function (assuming it exists)
        print(f"Generating 3D grid: L={L}, res={res}")
        # For demonstration, create a simple grid
        grid = generate_grid(L, res, 3)
        
        print(f"Grid shape: {grid.shape}")
        print(f"Grid range: x∈[{grid[:, 0].min():.2f}, {grid[:, 0].max():.2f}]")
        
        k = ka / self.sphere_radius
        
        # Calculate incident field
        print("Calculating incident field...")
        U_inc = self.incident_field_pytorch(grid, k, amplitude)
        
        # Calculate scattered field
        print("Calculating scattered field...")
        U_scat = self.scattered_field_pytorch(grid, ka, amplitude)
        
        # Total field
        U =  U_scat #+ U_inc
        
        # Set field to zero inside sphere (rigid boundary)
        grid_np = grid.cpu().numpy()
        r = np.sqrt(np.sum(grid_np**2, axis=1))
        inside_mask = r <= self.sphere_radius
        
        # Convert mask to torch and apply
        inside_mask_torch = torch.from_numpy(inside_mask).to(self.device)
        U[inside_mask_torch] = 0
        
        print(f"Field calculated! Shape: {U.shape}")
        print(f"Field range: {torch.abs(U).min():.3f} to {torch.abs(U).max():.3f}")
        
        return U, grid
    
    def reshape_to_3d(self, U_flat, res):
        """
        Reshape flattened field to 3D array for visualization
        
        Parameters:
        U_flat: flattened field array of shape (res^3,)
        res: resolution
        
        Returns:
        U_3d: 3D array of shape (res, res, res)
        """
        if isinstance(U_flat, torch.Tensor):
            U_flat = U_flat.cpu().numpy()
        
        return U_flat.reshape(res, res, res)
    
    def plot_field_slices_pytorch(self, U, grid, res, ka, L):
        """Plot field on 2D slices through 3D domain"""
        # Reshape to 3D for slicing
        U_3d = self.reshape_to_3d(U, res)
        
        # Create coordinate arrays
        line = np.linspace(-L, L, res)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Get middle indices
        mid = res // 2
        
        # Add circle to show sphere boundary
        circle = plt.Circle((0, 0), self.sphere_radius, fill=False, color='white', linewidth=2)
        
        # XY plane (z = 0)
        im1 = axes[0,0].imshow(np.real(U_3d[:, :, mid]).T, 
                               extent=[-L, L, -L, L], origin='lower', cmap='RdBu_r')
        axes[0,0].set_title('Re(U) - XY plane (z=0)')
        axes[0,0].set_xlabel('x/a')
        axes[0,0].set_ylabel('y/a')
        axes[0,0].add_patch(plt.Circle((0, 0), self.sphere_radius, fill=False, color='white', linewidth=2))
        plt.colorbar(im1, ax=axes[0,0])
        
        # XZ plane (y = 0)
        im2 = axes[0,1].imshow(np.real(U_3d[:, mid, :]).T, 
                               extent=[-L, L, -L, L], origin='lower', cmap='RdBu_r')
        axes[0,1].set_title('Re(U) - XZ plane (y=0)')
        axes[0,1].set_xlabel('x/a')
        axes[0,1].set_ylabel('z/a')
        axes[0,1].add_patch(plt.Circle((0, 0), self.sphere_radius, fill=False, color='white', linewidth=2))
        plt.colorbar(im2, ax=axes[0,1])
        
        # YZ plane (x = 0)
        im3 = axes[0,2].imshow(np.real(U_3d[mid, :, :]).T, 
                               extent=[-L, L, -L, L], origin='lower', cmap='RdBu_r')
        axes[0,2].set_title('Re(U) - YZ plane (x=0)')
        axes[0,2].set_xlabel('y/a')
        axes[0,2].set_ylabel('z/a')
        axes[0,2].add_patch(plt.Circle((0, 0), self.sphere_radius, fill=False, color='white', linewidth=2))
        plt.colorbar(im3, ax=axes[0,2])
        
        # Magnitude plots
        im4 = axes[1,0].imshow(np.abs(U_3d[:, :, mid]).T, 
                               extent=[-L, L, -L, L], origin='lower', cmap='plasma')
        axes[1,0].set_title('|U| - XY plane (z=0)')
        axes[1,0].set_xlabel('x/a')
        axes[1,0].set_ylabel('y/a')
        axes[1,0].add_patch(plt.Circle((0, 0), self.sphere_radius, fill=False, color='black', linewidth=2))
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].imshow(np.abs(U_3d[:, mid, :]).T, 
                               extent=[-L, L, -L, L], origin='lower', cmap='plasma')
        axes[1,1].set_title('|U| - XZ plane (y=0)')
        axes[1,1].set_xlabel('x/a')
        axes[1,1].set_ylabel('z/a')
        axes[1,1].add_patch(plt.Circle((0, 0), self.sphere_radius, fill=False, color='black', linewidth=2))
        plt.colorbar(im5, ax=axes[1,1])
        
        im6 = axes[1,2].imshow(np.abs(U_3d[mid, :, :]).T, 
                               extent=[-L, L, -L, L], origin='lower', cmap='plasma')
        axes[1,2].set_title('|U| - YZ plane (x=0)')
        axes[1,2].set_xlabel('y/a')
        axes[1,2].set_ylabel('z/a')
        axes[1,2].add_patch(plt.Circle((0, 0), self.sphere_radius, fill=False, color='black', linewidth=2))
        plt.colorbar(im6, ax=axes[1,2])
        
        plt.tight_layout()
        plt.suptitle(f'Acoustic Field Distribution (ka={ka}, incident: {self.incident_direction})', y=1.02)
        plt.show()



def evaluate_custom_estimation(model, trainer):
    '''
    For 3D Custom shape evaluation if solution on trainer.x_grid is given by trainer.solution
    '''
    x_grid = trainer.eval_grid
    target = torch.tensor(trainer.solution, device = trainer.device)          
    with torch.no_grad():
        model.eval()
        prediction = model(x_grid)

    nmse_real = nmse(prediction[:,0], target[:,0])
    nmse_imag = nmse(prediction[:,1], target[:,1])

    cos_sim_real = cosine_similarity(prediction[:,0], target[:,0])
    cos_sim_imag = cosine_similarity(prediction[:,1], target[:,1])
    
    print(f"NMSE Real: {nmse_real.item():.4f}, "
      f"NMSE Imag: {nmse_imag.item():.4f}")
    print(f"Cosine Similarity Real: {cos_sim_real.item()}, Cosine Similarity Imag: {cos_sim_imag.item()}")


########################################################
# 3D evaluation of the scattering estimation for HRTF
########################################################

class BoundaryIntegralSolver:
    def __init__(self, mesh_points, mesh_triangles, omega, c=343.0, device='cpu'):
        """
        Initialize the boundary integral solver for HRTF regularization.
        
        Parameters:
        - mesh_points: (N, 3) array of mesh vertex coordinates OR trimesh.Trimesh object
        - mesh_triangles: (M, 3) array of triangle vertex indices (ignored if mesh_points is trimesh)
        - omega: angular frequency
        - c: speed of sound
        - device: torch device for computations
        """
        self.device = torch.device(device)
        self.dtype = torch.float32
        
        # Handle both trimesh objects and traditional numpy arrays
        if hasattr(mesh_points, 'vertices'):  # It's a trimesh object
            self.mesh_points = torch.tensor(mesh_points.vertices, dtype=self.dtype, device=self.device)
            self.mesh_triangles = torch.tensor(mesh_points.faces, dtype=torch.long, device=self.device)
            self.trimesh = mesh_points
        else:  # Traditional numpy arrays
            self.mesh_points = torch.tensor(mesh_points, dtype=self.dtype, device=self.device)
            self.mesh_triangles = torch.tensor(mesh_triangles, dtype=torch.long, device=self.device)
            self.trimesh = None
            
        self.omega = omega
        self.c = c
        self.k = omega / c  # wave number
        
    def compute_triangle_properties(self):
        """Compute triangle centroids, areas, and normals."""
        # Use trimesh's built-in properties if available (more efficient and accurate)
        if self.trimesh is not None:
            centroids = torch.tensor(self.trimesh.triangles_center, dtype=self.dtype, device=self.device)
            areas = torch.tensor(self.trimesh.area_faces, dtype=self.dtype, device=self.device)
            normals = torch.tensor(self.trimesh.face_normals, dtype=self.dtype, device=self.device)
            return centroids, areas, normals
        
        # Fallback to manual computation using PyTorch
        triangles = self.mesh_points[self.mesh_triangles]  # Shape: (M, 3, 3)
        
        # Centroids
        centroids = torch.mean(triangles, dim=1)  # Shape: (M, 3)
        
        # Compute normals and areas using cross product
        v1 = triangles[:, 1] - triangles[:, 0]  # Shape: (M, 3)
        v2 = triangles[:, 2] - triangles[:, 0]  # Shape: (M, 3)
        normals = torch.cross(v1, v2, dim=1)    # Shape: (M, 3)
        areas = 0.5 * torch.norm(normals, dim=1)  # Shape: (M,)
        
        # Normalize normals
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-6)
        
        return centroids, areas, normals
    
    def green_function(self, x_ear, y_points):
        """
        Compute the free-space Green's function G(x_ear, y, ω).
        G(x, y, ω) = exp(ikr) / (4πr) where r = |x - y|
        
        Args:
            x_ear: (N_ears, 3) tensor of ear positions
            y_points: (M, 3) tensor of boundary points
            
        Returns:
            (N_ears, M) tensor of Green's function values
        """
        if x_ear.dim() == 1:
            x_ear = x_ear.unsqueeze(0)
        
        # Compute distances: (N_ears, M)
        distances = torch.cdist(x_ear, y_points)
        
        # Avoid division by zero for very small distances
        distances = torch.clamp(distances, min=1e-10)
        
        # Green's function (complex values)
        ikr = 1j * self.k * distances
        G = torch.exp(ikr) / (4 * np.pi * distances)
        
        return G.squeeze()
    
    def green_function_normal_derivative(self, x_ear, y_points, normals):
        """
        Compute the normal derivative of Green's function.
        ∂G/∂n = (ikr - 1) * exp(ikr) / (4πr²) * (r̂ · n̂)
        
        Args:
            x_ear: (N_ears, 3) tensor of ear positions
            y_points: (M, 3) tensor of boundary points
            normals: (M, 3) tensor of outward normals
            
        Returns:
            (N_ears, M) tensor of normal derivative values
        """
        if x_ear.dim() == 1:
            x_ear = x_ear.unsqueeze(0)
        
        N_ears = x_ear.shape[0]
        M = y_points.shape[0]
        
        # Expand dimensions for broadcasting
        x_ear_expanded = x_ear.unsqueeze(1)  # (N_ears, 1, 3)
        y_points_expanded = y_points.unsqueeze(0)  # (1, M, 3)
        normals_expanded = normals.unsqueeze(0)  # (1, M, 3)
        
        # Vector from y to x_ear: (N_ears, M, 3)
        r_vec = x_ear_expanded - y_points_expanded
        distances = torch.norm(r_vec, dim=2)  # (N_ears, M)
        distances = torch.clamp(distances, min=1e-10)
        
        # Unit vector: (N_ears, M, 3)
        r_hat = r_vec / distances.unsqueeze(2)
        
        # Dot product with normals: (N_ears, M)
        r_dot_n = torch.sum(r_hat * normals_expanded, dim=2)
        
        # Normal derivative
        ikr = 1j * self.k * distances
        dG_dn = (ikr - 1) * torch.exp(ikr) / (4 * np.pi * distances**2) * r_dot_n
        
        return dG_dn.squeeze()
    
    def compute_pressure_normal_derivative(self, y_points, normals, model):
        """
        Compute ∂P/∂n at boundary points using finite differences.
        
        Args:
            y_points: (M, 3) tensor of boundary points
            normals: (M, 3) tensor of outward normals
            model: neural network model that takes (M, 3) and returns (M, 2) [real, imag]
            
        Returns:
            (M,) tensor of complex normal derivatives
        """
        eps = 1e-6
        
        # Points slightly inside and outside the boundary
        y_in = y_points - eps * normals   # (M, 3)
        y_out = y_points + eps * normals  # (M, 3)
        
        # Compute pressure at these points using the model
        with torch.no_grad():
            P_in_complex = model(y_in)   # (M, 2) [real, imag]
            P_out_complex = model(y_out) # (M, 2) [real, imag]
        
        # Convert to complex tensors
        P_in = torch.complex(P_in_complex[:, 0], P_in_complex[:, 1])
        P_out = torch.complex(P_out_complex[:, 0], P_out_complex[:, 1])
        
        # Finite difference approximation
        dP_dn = (P_out - P_in) / (2 * eps)
        
        return dP_dn
    
    def inc_wave(self, x, config):
        """
        Compute incident plane wave.
        
        Args:
            x: (N, 3) tensor of observation points
            direction: (3,) tensor of wave direction
            
        Returns:
            (N, 2) tensor representing [real, imag] parts
        """

        x_source = config['source']
        r = torch.linalg.norm(x - x_source, dim = 1)   #(N,)
        r = r / config['scale']
        g = torch.exp(1j * self.k * r)/(4 * np.pi * r) # (N,) complex
        return torch.stack((g.real, g.imag), dim=-1)  # (N, 2)
    
    def compute_boundary_integral(self, sofa, model, config):
        """
        Compute the full boundary integral equation.
        
        Args:
            sofa: dictionary containing 'mic_positions' - (N_ears, 3) ear positions
            model: neural network that predicts pressure field
            config: dictionary containing 'direction' - (3,) wave direction
            
        Returns:
            P_tilde: (N_ears, 2) tensor [real, imag] pressure at ears
            debug_info: dictionary with intermediate computations
        """
        # Get ear positions as tensor
        x_ear = torch.tensor(sofa["mic_positions"], dtype=self.dtype, device=self.device)
        if x_ear.dim() == 1:
            x_ear = x_ear.unsqueeze(0)
        
        # Get wave direction as tensor
        direction = torch.tensor(config['direction'], dtype=self.dtype, device=self.device)
        
        # Compute triangle properties
        centroids, areas, normals = self.compute_triangle_properties()
        
        #Rescaling
        areas /= config['scale']**2
        x_ear /= config['scale']

        # Incident pressure at ear positions
        P_inc_ear = self.inc_wave(x_ear, config)  # (N_ears, 2)

        # Pressure at boundary points using the model
        centroids = centroids.to(self.dtype)
        P_boundary_complex, centroids = model(centroids, diff = True)  # (M, 2) [real, imag] #centroids * config['scale']
        P_boundary = torch.complex(P_boundary_complex[:, 0], P_boundary_complex[:, 1])  # (M,)
        
        # Normal derivative of pressure at boundary
        from utils import jacobian, gradient
        #dP_dn, _ = self.compute_pressure_normal_derivative(centroids, normals, model)  # (M,)
        grads_real = gradient(P_boundary_complex[..., 0], centroids) 
        grads_imag = gradient(P_boundary_complex[..., 1], centroids)

        # normal derivatives: dot product with normal vector
        dP_dn_real = torch.sum(grads_real * normals, dim=1, keepdim=True)
        dP_dn_imag = torch.sum(grads_imag * normals, dim=1, keepdim=True)
        dP_dn = dP_dn_real + 1j * dP_dn_imag
        dP_dn = dP_dn.squeeze(-1)
        # Green's function and its normal derivative
        centroids_scaled = centroids.detach().clone() / config['scale']
        G = self.green_function(x_ear, centroids_scaled)      # (N_ears, M)
        dG_dn = self.green_function_normal_derivative(x_ear, centroids_scaled, normals)  # (N_ears, M)
        
        # Boundary integral computation
        # integrand shape: (N_ears, M)
        integrand = P_boundary.unsqueeze(0) * dG_dn - G * dP_dn.unsqueeze(0)
        
        # Numerical integration (simple quadrature using triangle areas)
        # boundary_integral shape: (N_ears,)
        boundary_integral = torch.sum(integrand * areas.unsqueeze(0), dim=1)
        
        # Full formula: P_tilde = 2 * P_inc + 2 * boundary_integral
        P_inc_complex = torch.complex(P_inc_ear[:, 0], P_inc_ear[:, 1])  # (N_ears,)
        P_tilde_complex = 2 * P_inc_complex + 2 * boundary_integral  # (N_ears,)
        
        # Convert back to [real, imag] format
        P_tilde = torch.stack((P_tilde_complex.real, P_tilde_complex.imag), dim=-1)  # (N_ears, 2)
        
        return P_tilde, {
            'P_inc_ear': P_inc_ear,
            'boundary_integral': boundary_integral,
            'centroids': centroids,
            'areas': areas,
            'normals': normals,
            'P_boundary': P_boundary,
            'dP_dn': dP_dn,
            'G': G,
            'dG_dn': dG_dn
        }



def evaluate_hrtf(sofa, model, mesh_path, config):
    """Test the boundary integral solver with a simple sphere."""
    # Create a sphere mesh using trimesh
    mesh = trimesh.load_mesh(mesh_path, process=False)
    
    print(f"Mesh info:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.6f}")
    print(f"  Surface area: {mesh.area:.6f}")
    
    device = config['device']    
    
    # Initialize solver
    omega = 2 * np.pi * config['frequency'] #* 343 # 1 kHz

    x_ear = sofa['mic_positions']
    d_ear = torch.linalg.norm(0 * x_ear - config['source'], dim = 1)   # TO CHECK
    P_ref = config['p0']*torch.exp(1j * omega / config['celerity'] * d_ear) / (4 * np.pi * d_ear)
    P_ref = torch.stack([P_ref.real, P_ref.imag], dim=-1)
    P_sc = model(x_ear)
    print(f"\nResults:")
    print(f"P_sc shape: {P_sc.shape}")
    print(f"P_sc values:")
    for i, p in enumerate(P_sc):
        magnitude = torch.sqrt(p[0]**2 + p[1]**2)
        phase = torch.atan2(p[1], p[0])
        print(f"  Ear {i}: real={p[0]:.6f}, imag={p[1]:.6f}, mag={magnitude:.6f}, phase={phase:.6f}")

    P_est = P_sc + P_ref
    print(f"\nResults:")
    print(f"P_est shape: {P_est.shape}")
    print(f"P_est values:")
    for i, p in enumerate(P_est):
        magnitude = torch.sqrt(p[0]**2 + p[1]**2)
        phase = torch.atan2(p[1], p[0])
        print(f"  Ear {i}: real={p[0]:.6f}, imag={p[1]:.6f}, mag={magnitude:.6f}, phase={phase:.6f}")
    
    print(f"P_ref shape: {P_ref.shape}")
    print(f"P_ref values:")
    for i, p in enumerate(P_ref):
        magnitude = torch.sqrt(p[0]**2 + p[1]**2)
        phase = torch.atan2(p[1], p[0])
        print(f"  Ear {i}: real={p[0]:.6f}, imag={p[1]:.6f}, mag={magnitude:.6f}, phase={phase:.6f}")

    solver = BoundaryIntegralSolver(mesh, None, omega, device=device)
    
    # Compute the boundary integral
    P_tilde, debug_info = solver.compute_boundary_integral(sofa, model, config)
    
    print(f"\nResults:")
    print(f"P_tilde shape: {P_tilde.shape}")
    print(f"P_tilde values:")
    for i, p in enumerate(P_tilde):
        magnitude = torch.sqrt(p[0]**2 + p[1]**2)
        phase = torch.atan2(p[1], p[0])
        print(f"  Ear {i}: real={p[0]:.6f}, imag={p[1]:.6f}, mag={magnitude:.6f}, phase={phase:.6f}")
    
    hrtf = sofa[ 'hrtf']
    idf = torch.argmin(torch.abs(sofa['freqs'] - config['frequency']))
    print('idf', idf)
    sph = sofa['sources_positions'].cpu().numpy()
    # Degrees to radians
    az = np.deg2rad(sph[:, 0])  # θ
    el = np.deg2rad(sph[:, 1])  # φ

    # Spherical to Cartesian
    x =  np.cos(el) * np.cos(az)
    y =  np.cos(el) * np.sin(az)
    z =  np.sin(el)

    direction = np.stack([x, y, z], axis=-1) 
    direction = torch.tensor(direction).to(model.device)
    idd = torch.argmin(torch.linalg.norm(direction - config['direction'].T ,dim = 1))
    hrtf_df = hrtf[idd,:,idf,:]
    p_df = compl_mul(hrtf_df,P_ref)
    print('direction of reference :', direction[idd])
    print('reference frequency :', sofa['freqs'][idf])
    print('hrtf shape :', hrtf.shape)
    print('True htrf values : ',hrtf_df)
    for i, p in enumerate(p_df):
        magnitude = torch.sqrt(p[0]**2 + p[1]**2)
        phase = torch.atan2(p[1], p[0])
        print(f"  Ear {i}: real={p[0]:.6f}, imag={p[1]:.6f}, mag={magnitude:.6f}, phase={phase:.6f}")
    
    return solver, P_tilde, debug_info


def visualize_results(solver, debug_info):
    """Visualize the boundary integral computation results."""
    centroids = debug_info['centroids'].cpu().numpy()
    P_boundary = debug_info['P_boundary'].cpu()
    pressure_mag = torch.abs(P_boundary).numpy()
    
    fig = plt.figure(figsize=(12, 5))
    
    # Plot mesh with pressure field
    ax1 = fig.add_subplot(121, projection='3d')
    if solver.trimesh is not None:
        vertices = solver.trimesh.vertices
        faces = solver.trimesh.faces
        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                         triangles=faces, alpha=0.3, color='lightgray')
    
    # Color boundary points by pressure magnitude
    scatter = ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                         c=pressure_mag, s=20, cmap='viridis')
    plt.colorbar(scatter, ax=ax1, label='|P| magnitude')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Pressure Field on Boundary')
    
    # Plot Green's function magnitude
    ax2 = fig.add_subplot(122, projection='3d')
    G_mag = torch.abs(debug_info['G'][0]).cpu().numpy()  # First ear
    scatter2 = ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                          c=G_mag, s=20, cmap='plasma')
    plt.colorbar(scatter2, ax=ax2, label='|G| magnitude')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Green\'s Function Magnitude')
    
    plt.tight_layout()
    plt.show()




def evaluate_circle_estimation_d(model, config, R):
    L = config['L']
    res = config['res']
    x_grid = generate_grid(L, res, 2, device=model.device)
    mask = x_grid[:,0]**2 + x_grid[:,1]**2 > R**2
    print('mask shape :' ,mask.shape)
    k = 2 * np.pi * config['frequency'] / config['celerity']  # wavenumber
    with torch.no_grad():
        model.eval()
        theta_batch =  0. * np.pi * torch.rand((1)) #min(1, 2* (epoch - n_break) / config['epochs']) * torch.rand((16)) # 2 * np.pi - np.pi* torch.rand((10)) / 2 #- 1
        direction = torch.stack((torch.cos(theta_batch), torch.sin(theta_batch)), dim = 1).to(model.device)
        prediction = torch.zeros((res**2,2), device = config['device'])
        target = torch.zeros((res**2,2), device = config['device'], dtype = torch.double)
        prediction[mask,:] = model(x_grid[mask,:], direction)
        target[mask,:] = sound_hard_circle(config, x_grid[mask,:], R)

    nmse_real = nmse(prediction[:,0], target[:,0])
    nmse_imag = nmse(prediction[:,1], target[:,1])

    cos_sim_real = cosine_similarity(prediction[:,0], target[:,0])
    cos_sim_imag = cosine_similarity(prediction[:,1], target[:,1])

    print(f"NMSE Real: {nmse_real.item():.4f}, "
      f"NMSE Imag: {nmse_imag.item():.4f}")
    print(f"Cosine Similarity Real: {cos_sim_real.item()}, Cosine Similarity Imag: {cos_sim_imag.item()}")

    # Errors
    error = torch.abs(prediction - target).cpu().numpy()
    error_real = error[:,0].reshape(res, res)
    error_imag = error[:,1].reshape(res, res)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(error_real, extent=[-L, L, -L, L], origin='lower', cmap='inferno')
    axs[0].set_title('Absolute Error (Real Part)')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(error_imag, extent=[-L, L, -L, L], origin='lower', cmap='inferno')
    axs[1].set_title('Absolute Error (Imag Part)')
    fig.colorbar(im1, ax=axs[1])

    plt.suptitle('Error Between SIREN and True Scattered Field')
    plt.tight_layout()
    plt.show()







