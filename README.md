# Acoustic Scattering PINNs

This repository contains the implementation of methods described in the paper **"Physics-Informed Learning of Neural Scattering Fields Towards Measurement-Free Mesh-to-HRTF Estimation"**.
You can find an associated detailed report about parameters tuning, model explanation corresponding to my master thesis in master_thesis.pdf .

## Overview

This framework solves acoustic scattering problems using geometrical descriptions of scatterers and incident wave models through Physics-Informed Neural Networks (PINNs).

### Models


- **Reference PINN**: Solves acoustic scattering for a specific scatterer shape (mesh), incident wave direction, and frequency.
- **LoRA PINN**: Efficiently adapts a trained reference PINN to different incident wave directions using Low-Rank Adaptation.
- **PHISK** (PHysics-Informed Scattering hypernetworK): Solves scattering problems for all possible incident wave directions using a LoRA interpolation scheme.



## Installation

### Clone the Repository

```bash
git clone https://github.com/Dankradino/Acoustic-Scattering-PINN.git
cd Acoustic-Scattering-PINN
```

### Setup Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yaml -n phisk_env
conda activate phisk_env
```

**Important**: Verify that the **rtree** module is properly installed within **trimesh**. This optimization module is required for generating training points.



## Configuration

### File Structure

Configuration files define your scattering problem parameters:

- **2D problems**: `/config/2D/config.yaml`
- **3D problems**: `/config/3D/config_model_name.yaml` (model-specific)

Parameters have been tuned for efficient performance. We recommend testing with default parameters before modifications.

⚠️ **Note**: Avoid modifying hypernetwork parameters except for the number of epochs.

### Acoustic Parameters

Configure the following acoustic properties in your config file:

- **`frequency`**: Frequency of the incident wave (Hz)
- **`ref_direction`**: Reference direction for PINN training
- **`mode`**: Incident wave mode (default: `plane` wave)
- **`source`**: Source position (only required if `mode` is set to `source`)
- **`L`**: Domain size for scattered field estimation (recommended: ≤1.0; scale mesh instead of increasing L)

### Geometry Configuration

#### Supported Shapes

**2D**: Circle, ellipse, star (custom_shape can be easily adapted).
**3D**: Sphere, head model (SONICOM Dataset Engel and al. 2023), and custom meshes.

#### Using Custom Meshes

Custom 3D meshes are supported through Trimesh-compatible formats. For scattering fields wider than 2 meters, scale your mesh as demonstrated in `/mains/hrtf_training.py` rather than increasing the `L` parameter.

You can evaluate network performance for custom meshes using the scheme described in the directory `/mains/`.


## Training

Training scripts for all PHISK components are located in `/mains/`.

### Train PHISK from Scratch

After configuring your shape and parameters:

```bash
python mains/3D/phisk_training_from_scratch.py
```

### HRTF-Specific Training

A specialized training pipeline for the SONICOM dataset is available:

```bash
python mains/hrtf_training.py
```


## Project Structure

```
Acoustic-Scattering-PINN/
├── config/           # Configuration files
│   ├── 2D/
│   └── 3D/
├── mains/            # Training scripts
│   ├── 2D/
│   ├── 3D/
│   └── hrtf_training.py
├── environment.yaml  # Conda environment specification
└── README.md
```


## References

- Isaac Engel, Rapolas Daugintis, Thibault Vicente, Aidan Hogg, Johan Pauwels,
Arnaud Tournier, and Lorenzo Picinali. The SONICOM HRTF dataset. Journal
of the Audio Engineering Society, 71:241–253, 2023.

## Coming Soon

- Interactive Jupyter notebooks demonstrating model loading and testing.


## Contact

For questions or issues, please contact:
- Email: tancrede.martinez@telecom-paris.fr