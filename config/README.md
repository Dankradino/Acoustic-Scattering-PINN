# Config setup 

### Acoustical parameter of the scattering problem:
frequency: Frequency of the incident wave.
celerity: Sound celerity approximately 343 m/s.

### Problem setup:
res: Number of points of comparison with ground truth solution to the power of the dimension of the problem (e.g., 64 is 262144 evaluation points in 3D).
L: Computational domain length in [-L,L]. 

### Perfectly Matched Layer (PML) Setup:
pml_size: PML domain extension length so that the points lies in [-L-pml_size,L+pml_size]
pml_dampling: Damping factor of PML corresponds to a0 in the equation.

### Model setup
model: Model name in ['siren','rff','re_im']. We advise 'rff' for 3D processing.
device: Model device
input_dim: Input dimension (e.g., coordinate dimension).
hidden_dim: Hidden dimension
output_dim: Output dimesion (e.g., 2 in general for Real and Imaginary part).
num_layers: Number of hidden layers.
do_skip: Do_skipping inside layer (only for siren).

### Specific model parameter
first_omega: Omega0 of input layer of SIREN network.
hidden_omega: Omega0 of hidden layer of SIREN network.
sigma_fourier: Standard deviation 

### Training parameter for each step 'adam' for Adam phase, 'Fine' for L-BFGS phase or lora for 'LoRA' adaptation
lr: Learning rate.
epochs: Number of epochs.
n_scheduler: Scheduler number of iteration before update.
gamma: Gamma parameter for learning rate decrease of scheduler.
batch_size: Number of outer sampled points used for each epochs.
batch_boundary: Number of boundary sampled points used for each epochs.
keeping_time: Number of epochs where a same batch of points is used.
max_iter: L-BFGS m parameter.