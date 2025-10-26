# Training corresponding mains file documentation

This folder contains training main to solve scattering problem in both 2D and 3D.

---

## Overview

The training framework supports multiple training strategies:
1. **Reference Training**: Train a single reference PINN for one scattering configuration.
2. **LoRA Adaptation**: Efficiently adapt a trained model to new incident wave directions.
3. **PHISK Training**: Train a hypernetwork that generates direction-conditioned scattered fields.
4. **End-to-End Training**: Complete pipeline from scratch to PHISK.

---

## Command-Line Parameters
### Model Configuration
- **`--model`**: Neural network architecture type
  - Options: `'re_im'` (a SIREN for real part and a SIREN for imaginary part), `'rff'` (Random Fourier Features with QuadraticTanh activation (other custom activation functions can be used)), `'siren'` (SIREN network).
  - MUST BE SPECIFIED

### PHISK-Specific Parameters
- **`--J`**: Number of adapted incident directions for PHISK training
  - Determines how many LoRA-adapted models are used for LoRA interpolation in PHISK.
  - Recommanded: 4-32 directions

### Geometry Configuration
- **`--mesh_path`**: Path to custom mesh file for 3D scatterers
  - Format: Trimesh-compatible format (.obj, .stl, etc.)
  - Leave empty to use built-in geometric spherical shape (in 2D other example are provided, feel free to test).
  - Example: `'./meshes/head.stl'`

### Training Mode Options
- **`--preload`**: Bool to continue training from a checkpoints `save_dir`.
  - Only available for: `reference_training.py` and `phisk_training.py`
  - Enables transfer learning or interrupted training resumption

- **`--retrain`**: Bool which must be set to True if no reference PINN as been trained for the problem.
  - Only available for: `lora_adaptation.py`

### Output Directory Configuration
- **`--save_dir`**: Base directory for saving reference PINN models
  - Final path: `{save_dir}{model_name}.pth`

- **`--lora_dir`**: Base directory for saving LoRA-adapted weights
  - Final path: `{lora_dir}/{adapted_direction}.pth`

- **`--hsave_dir`**: Base directory for saving PHISK hypernetwork models
  - Final path: `{hsave_dir}/{model_name}.pth`

## Available Training Scripts

### 1. `reference_training.py`
**Purpose**: Train a single reference PINN for one scattering configuration (specific direction `d` and frequency `f`)

**Example usage**:
```bash
# Train from scratch
python reference_training.py --model rff --save_dir ./checkpoints/3D/scattering/

# Continue training from checkpoint
python reference_training.py --model siren --preload True --save_dir ./mu_old_checkpoints_dir/
```

**Requirements**:
- Model name

**Output**: 
- Trained reference PINN model saved in `{save_dir}/{model_name}.pth`
- Training logs and NMSE/CosDist if available

---

### 2. `lora_adaptation.py`
**Purpose**: Adapt a reference PINN to new incident wave directions using Low-Rank Adaptation (LoRA)

**Description**: 
- LoRA enables efficient fine-tuning by adding trainable low-rank matrices to frozen model weights
- Much faster than training from scratch for each new direction
- Can either load an existing reference model or train one first

**Example usage**:
```bash
# Adapt existing reference model
python lora_adaptation.py --model rff --J 12

# Train reference model and then adapt
python lora_adaptation.py --model rff --retrain True --J 16 --lora_dir ./checkpoints/hrtf/lora/
```
**Requirements**:
- Model name

**Output**:
- LoRA weight deltas saved in `{lora_dir}{direction}.pth`
- One set of weights per adapted direction

---

### 3. `phisk_training.py`
**Purpose**: Train a PHISK (PHysics-Informed Scattering hypernetworK) that generates direction-conditioned models

**Description**:
- Requires: A reference PINN and multiple LoRA-adapted weights
- PHISK learns to correct LoRA interpolation between adapted directions
- Enables evaluation at arbitrary incident directions without retraining

**Example usage**:
```bash
# Train PHISK from existing reference + LoRA weights
python phisk_training.py --model rff --save_dir ./checkpoints/3D/scattering/ --lora_dir ./checkpoints/3D/lora/ --hsave_dir ./checkpoints/3D/phisk/ 

# Continue PHISK training from checkpoint
python phisk_training.py --model siren --preload True
```

**Requirements**:
- Model name
- Previously trained reference PINN and LoRA adapted weights

**Output**:
- PHISK hypernetwork saved in `{hsave_dir}{model_name}.pth`

---

### 4. `phisk_training_from_scratch.py`
**Purpose**: Execute the complete training pipeline in one script

**Description**:
Automates the full workflow:
1. Train reference PINN for one direction
2. Perform LoRA adaptation for `J` directions
3. Train PHISK hypernetwork
4. Save all intermediate and final models

**Example usage**:
```bash
python phisk_training_from_scratch.py --J 20 --model rff --save_dir ./checkpoints/hrtf/scattering/ --lora_dir ./checkpoints/hrtf/lora/ --hsave_dir ./checkpoints/hrtf/phisk/
```

**Requirements**:
- Model name

**Output**:
- Reference model in `{save_dir}{model_name}.pth`
- LoRA adaptations in `{lora_dir}`
- Final PHISK model in `{hsave_dir}{model_name}.pth`

---


### 5. `hrtf_training.py`
**Purpose**: Exemple of one frequency HRTF PHISK training with a custom save id for the experiment in SONICOM experiment setup.

**Description**:
Automates the full workflow:
1. Train reference PINN for one direction
2. Perform LoRA adaptation for `J` directions
3. Train PHISK hypernetwork
4. Save all intermediate and final models

**Example usage**:
```bash
python phisk_training_from_scratch.py --J 24 --model rff --subject_id P0002 --freq_idx 4
```

**Requirements**:
- Subject id
- Frequency index for sofa comparison

**Output**:
- Reference model in `f"checkpoints/hrtf/{subject_id}_{config['frequency']}/scattering/"{model_name}.pth`
- LoRA adaptations in `f"checkpoints/hrtf/{subject_id}_{config['frequency']}/lora/"{model_name}.pth`
- Final PHISK model in `f"checkpoints/hrtf/{subject_id}_{config['frequency']}/scattering/"{model_name}.pth`

---

## Custom Mesh Processing

### 2D Custom Shapes

**Built-in shapes**: Circle, square, ellipse, star (see config documentation)

**Adding custom shapes**:
1. Define boundary points as a numpy array of shape `(N, 2)` where N is the number of boundary points.
2. Compute outward normal vectors at each boundary point, shape `(N, 2)`.
3. Integrate into the shape module or into boundary and normals definition.

**Example**:
```python
# In your geometry definition
boundary_points = np.array([[x1, y1], [x2, y2], ...])  # Shape: (N, 2)
normals = np.array([[nx1, ny1], [nx2, ny2], ...])      # Shape: (N, 2)
```

**Note**: 
- Ground truth evaluation for custom 2D shapes must be implemented manually.
- See 3D evaluation approach below for reference.

---

### 3D Custom Meshes

**Mesh format**: Trimesh-compatible formats (.obj, .stl, .ply, etc.)

**Usage**:
```bash
python reference_training.py --mesh_path ./meshes/head.obj
```

**Ground truth evaluation**:

For validation and comparison, provide two files in `./custom_mesh/`:

1. **`evaluation_points.npy`**: 
   - Numpy array of shape `(M, 3)` containing [x, y, z] coordinates
   - Points where the scattered field should be evaluated
```python
   # Example: Create evaluation grid
   evaluation_points = np.array([[x1, y1, z1], [x2, y2, z2], ...])
   np.save('./custom_mesh/evaluation_points.npy', evaluation_points)
```

2. **`scattered_field.npy`**:
   - Numpy array of shape `(M, 2)` containing [Real(P_s), Imag(P_s)]
   - Ground truth scattered pressure field at evaluation points
   - Can be obtained from: measurements, FEM/BEM simulations, or analytical solutions
```python
   # Example: Save reference solution
   scattered_field = np.array([[real_p1, imag_p1], [real_p2, imag_p2], ...])
   np.save('./custom_mesh/scattered_field.npy', scattered_field)
```

**Workflow**:
1. Generate or obtain 3D mesh of scatterer
2. Run FEM/BEM solver or measurements to get reference solution
3. Save evaluation points and scattered field as `.npy` files
4. Place files in `./custom_mesh/` directory
5. Run training with `--mesh_path` pointing to your mesh

---
