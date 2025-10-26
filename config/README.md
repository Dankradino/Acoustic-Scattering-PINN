# Configuration File Documentation

This configuration file defines parameters for solving acoustic scattering problems using PHISK architecture.



# For config.yaml configuring reference PINN and LoRA adaptation training.

### Acoustical Properties
- **`frequency`**: Frequency of the incident acoustic wave [Hz].
- **`celerity`**: Speed of sound in the medium [m/s] (default: 343 m/s for air at 20°C).

---

## Spatial domain Configuration

### Computational Domain
- **`res`**: Resolution per dimension for evaluation. Total evaluation points is res^d where d is the spatial dimension
  - Example: res=64 gives 64²=4,096 points in 2D or 64³=262,144 points in 3D
- **`L`**: Half-width of the computational domain [m]. Domain extends from [-L, L] in each dimension
- **`ref_direction`**: Direction for the reference PINN simulation.

### Perfectly Matched Layer (PML)
The PML absorbs outgoing waves at domain boundaries to simulate an infinite domain.

- **`pml_size`**: PML layer thickness [m]. Extended domain becomes [-L-pml_size, L+pml_size]
- **`pml_dampling`**: PML damping coefficient (a₀ in the damping profile)
  - Higher values increase wave absorption at boundaries.
  - Recommanded range: 1.0 - 10.0 (may increase the baseline parameter for higher frequencies).

---

## Scatterer Geometry

Define the shape and position of the scattering object using `mesh_param` if the selected mesh is not custom (e.g., not coming from a mesh file such as an head):

### Common Parameters
- **`center`**: [x, y] or [x, y, z] coordinates of the scatterer center [m]
- **`rotation`**: Rotation angle applied to the shape [radians]

### Shape-Specific Parameters

**Circle/Sphere:**
- **`r`**: Radius [m]

**Square/Cube:**
- **`length`**: Edge length [m]

**Star:**
- **`num_points`**: Number of star points/corners
- **`inner_radius`**: Distance from center to inner vertices [m]
- **`outer_radius`**: Distance from center to outer vertices [m]
- **`epsilon`**: Smoothing parameter for rounded corners [m]

**Ellipse/Ellipsoid:**
- **`ax`**: Semi-axis length in x-direction [m]
- **`ay`**: Semi-axis length in y-direction [m]

---

## Neural Network Model Configuration

### Model Architecture
- **`model`**: Model type, choose from:
  - `'siren'`: Sinusoidal Representation Networks SIREN.
  - `'rff'`: Random Fourier Features (recommended for 3D problems - more stable)
  - `'re_im'`: Separate real/imaginary SIREN networks.
  
- **`input_dim`**: Input dimension (2 for 2D problems, 3 for 3D problems)
- **`hidden_dim`**: Number of neurons per hidden layer (recommanded: 64 to 512)
- **`output_dim`**: Output dimension 
- **`num_layers`**: Number of hidden layers (recommanded: 3 to 4)
- **`do_skip`**: Enable skip connections between layers (boolean, only for SIREN model)

### Model-Specific Hyperparameters

**SIREN-specific:**
- **`first_omega`**: Frequency parameter ω₀ for the input layer (recommanded: 10.0 to 30.0).
- **`hidden_omega`**: Frequency parameter ω₀ for hidden layers (recommanded: 30.0).

**RFF-specific:**
- `'rff'`: Number of Random Fourier Features (recommended: 64 to 512)
- **`sigma_fourier`**: Standard deviation of Fourier feature sampling distribution (recommanded : 0.1 to 10.0).

---

## Training Configuration

The training process typically consists of multiple phases with different optimizers and strategies.

### Phase Identifiers
Training phases are specified by their type:
- **`'adam'`**: Adam optimizer phase (initial training).
- **`'fine'`**: L-BFGS optimizer phase.
- **`'lora'`**: Low-Rank Adaptation phase (for direction adaptation).

### Training Parameters (per phase)

**Optimization:**
- **`lr`**: Learning rate
  - Adam phase: Recommanded range from 1e-5 to 1e-3
  - L-BFGS phase: Recommanded range from 0.05 to 1.0
  
- **`epochs`**: Number of training epochs for this phase

**Learning Rate Scheduling:**
- **`n_scheduler`**: Number of epochs between learning rate updates (only usefull for Adam)
- **`gamma`**: Multiplicative factor for learning rate decay (e.g., 0.9 means 10% reduction)

**Batch Configuration:**
- **`batch_size`**: Number of collocation points sampled in the domain per batch.
- **`batch_boundary`**: Number of boundary points sampled per batch.
- **`keeping_time`**: Number of epochs to reuse the same batch of sampled points.

**L-BFGS Specific:**
- **`max_iter`**: Maximum iterations per L-BFGS step; also controls the history size (m) for the quasi-Newton approximation (recommanded : 20-30)



# Additional information for hconfig.yaml configuring PHISK Training.
**PHISK specific parameters**
- **`r`** Rank of LoRA adapted (only used in phisk_training_from_scratch.py). Set as 12 as explained in the report.
- **`hidden_dims`** : Structure of corrective hypernetwork hidden layers.
- **`num_fourier_features`** Number of hypernetwork Random Fourier Features.