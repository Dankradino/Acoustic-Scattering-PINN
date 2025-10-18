import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .activation import *

'''
This module implements the differents components of PHISK
'''

    
class DirectionInterpolationModule(nn.Module):
    """
    Attention mechanism to compute weights for LoRA ensemble based on target direction
    """
    def __init__(self, direction_dim=2, num_loras=None, hidden_dim=64, lora_directions = None, T = None):
        super().__init__()
        if T is None:
            T = torch.tensor([2 / lora_directions.shape[0]], dtype=torch.float32)
        else :
            if not isinstance(T, torch.Tensor):
                T = torch.tensor([T], dtype=torch.float32)
            else:
                T = T.to(dtype=torch.float32)

        self.direction_dim = direction_dim
        self.num_loras = num_loras
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.directions = lora_directions
        self.register_buffer("T", T) 

    def forward(self, target_direction):
        """
        Args:
            target_direction: [batch_size, direction_dim] or [direction_dim]
        Returns:
            attention_weights: [batch_size, num_loras] or [num_loras]
        """
        # Ensure shape is [batch_size, d]
        if target_direction.ndim == 1:
            target_direction = target_direction.unsqueeze(0)

        # Normalize input
        target_direction = target_direction / target_direction.norm(dim=-1, keepdim=True)  # [B, d]

        # Cosine similarity with all LoRA directions
        # [B, d] @ [d, n_loras] = [B, n_loras]
        sims = torch.matmul(target_direction, self.directions.T)

        # Optional sharpening — controls how “soft” the weights are
        weights = F.softmax(sims / self.T, dim=-1)  # [B, n_loras]
        #print(weights)
        return weights.squeeze(0) if weights.shape[0] == 1 else weights

class ContinuousDirectionHyperNetworkCorrector(nn.Module):
    """
    Hypernetwork for continuous direction refinement
    """
    def __init__(self, direction_dim=2, target_param_size=None, hidden_dims=[128, 256, 512], num_fourier_features = 64):  
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_fourier_features = num_fourier_features
        self.sigma_fourier = 1.
        self.direction_dim = direction_dim
        self.target_param_size = target_param_size

        self.B = self.sigma_fourier * torch.randn(
            (self.direction_dim, self.num_fourier_features), 
            device=self.device
        )
    
        
        # Build the hypernetwork
        layers = []
        prev_dim = self.num_fourier_features * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                QuadraticTanh(alpha = 0.1),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for parameter deltas)
        layers.append(nn.Linear(prev_dim, target_param_size))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize output layer with small weights for stability
        nn.init.xavier_normal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, direction):
        """
        Args:
            direction: [batch_size, direction_dim] or [direction_dim]
        Returns:
            parameter_deltas: [batch_size, target_param_size] or [target_param_size]
        """
        direction = self.fourier_feature_mapping(direction)
        return self.net(direction)
    
        
    def fourier_feature_mapping(self, x):
        """Apply Fourier feature mapping to input coordinates"""
        #print(x.shape, self.B.shape)
        return torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi*x @ self.B)], dim=-1)
    


import os
import glob
from collections import OrderedDict
import re

class PhiskModule:
    def __init__(self):
        # New components for continuous direction control
        self.lora_states = {}
        self.lora_directions = []
        self.direction_interpolation = None
        self.continuous_hypernetwork = None
        self.param_shapes = None


    def load_lora_ensemble(self, lora_dir):
        """
        Load LoRA ensemble from directory
        """
        print(f"Loading LoRA ensemble from {lora_dir}")
        
        # Find all .pth files
        pattern = os.path.join(lora_dir, "*.pth")
        weight_files = glob.glob(pattern)
        
        if not weight_files:
            raise ValueError(f"No .pth weight files found in {lora_dir}")
        
        print(f"Found {len(weight_files)} LoRA weight files")
        
        # Load LoRA states and extract directions
        self.lora_states = {}
        self.lora_directions = []
        
        for weight_file in weight_files:
            filename = os.path.basename(weight_file)
            print(f"Loading {filename}")
            
            # Load LoRA state
            lora_state = torch.load(weight_file, map_location=self.device)
            
            # Debug structure on first load
            if len(self.lora_states) == 0:
                print("Debug: LoRA state structure:")
                self._print_dict_structure(lora_state, max_depth=3)
            
            # Parse direction from filename
            direction = self._parse_direction_from_filename(filename)
            
            # Store
            direction_key = tuple(direction.cpu().numpy())
            self.lora_states[direction_key] = lora_state
            self.lora_directions.append(direction)
            
            print(f"Direction: {direction.cpu().numpy()}")
        
        # Convert directions to tensor
        self.lora_directions = torch.stack(self.lora_directions).to(self.device)
        
        # Initialize interpolation mechanism
        self.direction_interpolation = DirectionInterpolationModule(
            direction_dim=self.dim,
            num_loras=len(self.lora_states),
            hidden_dim=64,
            lora_directions= self.lora_directions,
        ).to(self.device)
        
        # Get parameter shapes from first LoRA state
        first_lora = list(self.lora_states.values())[0]
        self.param_shapes = self._get_param_shapes_from_lora_state(first_lora)
        
        print("Debug: Parameter shapes found:")
        for key, shape in list(self.param_shapes.items())[:5]:
            print(f"  {key}: {shape}")
        if len(self.param_shapes) > 5:
            print(f"  ... and {len(self.param_shapes) - 5} more")
        
        # Calculate total parameter size for hypernetwork
        total_param_size = sum(np.prod(shape) for shape in self.param_shapes.values())
        
        # Initialize continuous hypernetwork
        self.continuous_hypernetwork = ContinuousDirectionHyperNetworkCorrector(
            direction_dim=self.dim,
            target_param_size=total_param_size,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        print(f"Initialized interpolation mechanism for {len(self.lora_states)} LoRAs")
        print(f"Total parameter size: {total_param_size}")

    def _print_dict_structure(self, d, indent=0, max_depth=3):
        """Print the structure of a nested dictionary for debugging"""
        if indent > max_depth:
            print("  " * indent + "...")
            return
            
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}: dict with {len(value)} keys")
                self._print_dict_structure(value, indent + 1, max_depth)
            elif hasattr(value, 'shape'):
                print("  " * indent + f"{key}: tensor {value.shape}")
            else:
                print("  " * indent + f"{key}: {type(value)} = {value}")
            
            if indent == 0 and len(d) > 5:
                remaining = len(d) - list(d.keys()).index(key) - 1
                if remaining > 0:
                    print("  " * indent + f"... and {remaining} more keys")
                break
        
    def _get_param_shapes_from_lora_state(self, lora_state):
        """Extract parameter shapes from LoRA state dictionary"""
        param_shapes = OrderedDict()
        
        def extract_shapes_recursive(state_dict, prefix=""):
            for key, value in state_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Check if this is a layer dict with LoRA parameters
                    if any(lora_key in value for lora_key in ['lora_A', 'lora_B']):
                        # Extract LoRA parameter shapes from this layer
                        for lora_param in ['lora_A', 'lora_B']:
                            if lora_param in value and hasattr(value[lora_param], 'shape'):
                                param_shapes[f"{full_key}.{lora_param}"] = value[lora_param].shape
                        
                        # Only add bias if it actually exists and is a tensor
                        if 'bias' in value and hasattr(value['bias'], 'shape'):
                            param_shapes[f"{full_key}.bias"] = value['bias'].shape
                    else:
                        # Recursive case: nested dictionary
                        extract_shapes_recursive(value, full_key)
                elif hasattr(value, 'shape'):
                    # Direct tensor
                    param_shapes[full_key] = value.shape
                elif isinstance(value, (int, float)):
                    # Skip scalar values like alpha
                    continue
        
        extract_shapes_recursive(lora_state)
        return param_shapes
    
    def _get_all_param_keys(self, nested_dict, prefix=""):
        """Get all LoRA parameter keys from nested dictionary structure"""
        keys = []
        for key, value in nested_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                # Check if this dict contains LoRA parameters directly
                if any(lora_key in value for lora_key in ['lora_A', 'lora_B']):
                    # This is a layer dict containing LoRA parameters
                    for lora_param in ['lora_A', 'lora_B']:
                        if lora_param in value:
                            keys.append(f"{full_key}.{lora_param}")
                    # Only add bias if it exists and is a tensor
                    if 'bias' in value and hasattr(value['bias'], 'shape'):
                        keys.append(f"{full_key}.bias")
                else:
                    # Recurse into nested dicts
                    keys.extend(self._get_all_param_keys(value, full_key))
            elif hasattr(value, 'shape'):
                # Direct tensor
                keys.append(full_key)
        return keys
    

    def _get_tensor_from_nested_dict(self, nested_dict, key_path):
        """Get tensor from nested dictionary using key path with multiple fallback strategies"""
        # Strategy 1: Direct key lookup (flat structure)
        if key_path in nested_dict:
            return nested_dict[key_path]
        
        # Strategy 2: Two-level lookup (module.param)
        try:
            module_key, param_key = key_path.rsplit('.', 1)
            if module_key in nested_dict and isinstance(nested_dict[module_key], dict):
                if param_key in nested_dict[module_key]:
                    return nested_dict[module_key][param_key]
        except (ValueError, KeyError):
            pass
        
        # Strategy 3: Full path traversal (nested structure)
        try:
            keys = key_path.split('.')
            current = nested_dict
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            pass
        
        # Strategy 4: Check if this is a bias parameter that doesn't exist - return None
        if key_path.endswith('.bias'):
            #print(f"[INFO] Bias parameter '{key_path}' not found in LoRA state - this is normal if bias wasn't adapted")
            return None
        
        # If all strategies fail, provide debugging info
        print(f"[ERROR] Could not resolve key path: '{key_path}'")
        print(f"Available top-level keys: {list(nested_dict.keys())[:10]}")
        
        # Try to find similar keys
        all_keys = self._get_all_param_keys(nested_dict)
        similar_keys = [k for k in all_keys if key_path.split('.')[-1] in k]
        if similar_keys:
            print(f"Similar keys found: {similar_keys[:5]}")
        
        raise KeyError(f"Key path '{key_path}' not found in nested dict")

    def _set_tensor_in_nested_dict(self, nested_dict, key_path, tensor):
        """Set tensor in nested dictionary using dot-separated key path"""
        keys = key_path.split('.')
        current = nested_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = tensor

    def _parse_direction_from_filename(self, filename):
        """Parse direction from filename like '+1+0.pth' -> [1.0, 0.0], including scientific notation"""
        name = filename.replace('.pth', '')
        
        try:
            # Match signed floats, including scientific notation (e.g. +6.12e-17)
            float_strings = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', name)
            coords = [float(s) for s in float_strings]
            
            if not coords:
                raise ValueError("No coordinates found")
            
            return torch.tensor(coords, dtype=torch.float32, device=self.device)
        
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse direction from filename '{filename}': {e}")
            return torch.zeros(2, device=self.device)

    def _compute_lora_ensemble(self, interpolation_weights):
        """Compute weighted combination of LoRA parameters"""
        ensemble_params = {}
        
        # Get all parameter keys from first LoRA state
        first_lora = list(self.lora_states.values())[0]
        all_param_keys = self._get_all_param_keys(first_lora)
        
        # Initialize with zeros using proper nested structure
        for param_key in all_param_keys:
            param_tensor = self._get_tensor_from_nested_dict(first_lora, param_key)
            if param_tensor is not None:  # Skip None values (missing bias parameters)
                param_shape = param_tensor.shape
                self._set_tensor_in_nested_dict(ensemble_params, param_key, 
                                            torch.zeros(param_shape, device=self.device))
        
        # Weighted sum
        for i, (direction_key, lora_state) in enumerate(self.lora_states.items()):
            weight = interpolation_weights[i]
            for param_key in all_param_keys:
                param_tensor = self._get_tensor_from_nested_dict(lora_state, param_key)
                if param_tensor is not None:  # Skip None values
                    current_tensor = self._get_tensor_from_nested_dict(ensemble_params, param_key)
                    if current_tensor is not None:  # Make sure we have something to add to
                        updated_tensor = current_tensor + weight * param_tensor
                        self._set_tensor_in_nested_dict(ensemble_params, param_key, updated_tensor)

        for param_key in all_param_keys:
            if param_key.endswith('.bias'):
                # Check if ANY LoRA has this bias parameter
                any_lora_has_bias = False
                bias_shape = None
                
                for lora_state in self.lora_states.values():
                    bias_tensor = self._get_tensor_from_nested_dict(lora_state, param_key)
                    if bias_tensor is not None:
                        any_lora_has_bias = True
                        bias_shape = bias_tensor.shape
                        break
                
                if any_lora_has_bias:
                    # Initialize with zeros
                    self._set_tensor_in_nested_dict(ensemble_params, param_key, 
                                                torch.zeros(bias_shape, device=self.device))
                    
                    # Weighted sum of bias terms
                    for i, (direction_key, lora_state) in enumerate(self.lora_states.items()):
                        weight = interpolation_weights[i]
                        bias_tensor = self._get_tensor_from_nested_dict(lora_state, param_key)
                        if bias_tensor is not None:  # Only add if this LoRA has bias
                            current_bias = self._get_tensor_from_nested_dict(ensemble_params, param_key)
                            updated_bias = current_bias + weight * bias_tensor
                            self._set_tensor_in_nested_dict(ensemble_params, param_key, updated_bias)
            
        return ensemble_params

    def get_continuous_direction_params(self, target_direction):
        """
        Compute parameters for continuous direction using ensemble + hypernetwork
        """
        if not hasattr(self, 'direction_interpolation') or self.direction_interpolation is None:
            raise ValueError("LoRA ensemble not loaded. Call load_lora_ensemble() first.")
        
        target_direction = target_direction.to(self.device)
        
        # Get interpolation weights for LoRA ensemble
        interpolation_weights = self.direction_interpolation(target_direction)
        
        # Compute weighted combination of LoRA parameters
        ensemble_params = self._compute_lora_ensemble(interpolation_weights)
        
        # Get continuous refinement from hypernetwork
        continuous_deltas = self.continuous_hypernetwork(target_direction)
        
        # Convert deltas back to parameter dict format
        delta_params = self._vector_to_param_dict(continuous_deltas, self.param_shapes)
        
        # Combine ensemble and continuous refinement
        combined_params = {}
        all_param_keys = self._get_all_param_keys(ensemble_params)
        
        for param_key in all_param_keys:
            ensemble_tensor = self._get_tensor_from_nested_dict(ensemble_params, param_key)
            delta_tensor = self._get_tensor_from_nested_dict(delta_params, param_key)
            
            # Only combine if both tensors exist
            if ensemble_tensor is not None and delta_tensor is not None:
                combined_tensor = ensemble_tensor + delta_tensor
                self._set_tensor_in_nested_dict(combined_params, param_key, combined_tensor)
            elif ensemble_tensor is not None:
                # Use ensemble tensor if delta doesn't exist
                self._set_tensor_in_nested_dict(combined_params, param_key, ensemble_tensor)
            elif delta_tensor is not None:
                # Use delta tensor if ensemble doesn't exist
                self._set_tensor_in_nested_dict(combined_params, param_key, delta_tensor)
        
        return combined_params, interpolation_weights

    def _vector_to_param_dict(self, param_vector, param_shapes):
        """Convert flat parameter vector back to nested dictionary format"""
        param_dict = {}
        start_idx = 0
        
        for param_key, shape in param_shapes.items():
            param_size = np.prod(shape)
            end_idx = start_idx + param_size
            
            param_data = param_vector[start_idx:end_idx].reshape(shape)
            self._set_tensor_in_nested_dict(param_dict, param_key, param_data)
            
            start_idx = end_idx
        
        return param_dict

    def forward_with_lora_params(self, coords, lora_params, diff=False):
        """Forward pass through base network with LoRA parameters applied"""
        coords = coords.clone().detach().requires_grad_(diff)
        
        # Access the network structure
        if hasattr(self.base_network, 'siren'):
            if hasattr(self.base_network.siren, 'net'):
                siren_net = self.base_network.siren.net
            else:
                siren_net = self.base_network.siren
        elif hasattr(self.base_network, 'net'):
            siren_net = self.base_network.net
        else:
            raise AttributeError("Cannot find network structure in base_network")
        
        # Get all available LoRA parameter keys
        all_keys = self._get_all_param_keys(lora_params)
        layer_keys = [key for key in all_keys if 'lora_A' in key]
        
        if hasattr(self.base_network, 'fourier_feature_mapping'):
            out = self.base_network.fourier_feature_mapping(coords)
        else:
            out = coords
        
        # Process each layer
        for layer_idx in range(len(siren_net)):
            layer = siren_net[layer_idx]
            # Get the actual linear layer
            if hasattr(layer, 'linear'):
                linear_layer = layer.linear
            else:
                linear_layer = layer
            
            # Find LoRA parameters for this layer
            layer_pattern = f".{layer_idx}."
            matching_keys = [key for key in layer_keys if layer_pattern in key]
            #print(matching_keys)
            if matching_keys:
                # Apply LoRA adaptation
                base_key = matching_keys[0].replace('.lora_A', '')
                try:
                    lora_A = self._get_tensor_from_nested_dict(lora_params, f"{base_key}.lora_A")
                    lora_B = self._get_tensor_from_nested_dict(lora_params, f"{base_key}.lora_B")
                    
                    # Compute weight delta
                    weight_delta = (lora_B @ lora_A)
                    adapted_weight = linear_layer.weight + weight_delta
                    
                    # Handle bias
                    adapted_bias = linear_layer.bias
                    lora_bias = self._get_tensor_from_nested_dict(lora_params, f"{base_key}.bias")
                    if lora_bias is not None:
                        print('Nothing')
                        adapted_bias = linear_layer.bias + lora_bias

                    #print(out.shape, adapted_weight.shape, adapted_bias.shape)
                    # Apply adapted layer
                    out = F.linear(out, adapted_weight, adapted_bias)
                    
                except KeyError as e:
                    print(f"Warning: Could not apply LoRA to layer {layer_idx}: {e}")
                    # Fallback to original layer
                    out = linear_layer(out)
            else:
                # Use original layer
                out = linear_layer(out)
        
        if diff:
            return out, coords
        return out
    
    def create_forward_method(self, params):
        """Create forward method with captured parameters"""
        def forward_method(coords):
            return self.forward_with_lora_params(coords, params, diff=True)
        return forward_method

    def save_current(self, filename):
        torch.save({
            'interpolation_state': self.direction_interpolation.T, #self.direction_interpolation.state_dict(),   #Save only temperature but if you want to change simple interpolation by a neural interpolation it can be adaapted
            'hypernetwork_state': self.continuous_hypernetwork.state_dict(),
            'fourier_features_B': self.continuous_hypernetwork.B,  # ADD THIS
            'lora_directions': self.lora_directions,
            'param_shapes': self.param_shapes,
            'lora_states': self.lora_states
        }, filename)

    def load_hypernetwork_checkpoint(self, checkpoint_path):
        """
        Load saved hypernetwork, interpolation mechanism, and related data from checkpoint
        
        Args:
            checkpoint_path: Path to the saved checkpoint file
        """
        print(f"Loading hypernetwork checkpoint from {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        print(checkpoint.keys())  # to see all keys, including your RFF matrices
        
        # Extract saved data
        
        interpolation_state = checkpoint['interpolation_state']
        #T = checkpoint.get('interpolation_state', None)
        #print(T)
        hypernetwork_state = checkpoint['hypernetwork_state']
        self.lora_directions = checkpoint['lora_directions'].to(self.device)
        #print(self.lora_directions)
        self.param_shapes = checkpoint['param_shapes']
        
        # Initialize the interpolation mechanism and hypernetwork with correct dimensions
        num_loras = len(self.lora_directions)
        total_param_size = sum(np.prod(shape) for shape in self.param_shapes.values())
        
        # Initialize interpolation mechanism
        self.direction_interpolation = DirectionInterpolationModule(
            direction_dim = self.dim,
            num_loras = num_loras,
            hidden_dim = 64,
            lora_directions=self.lora_directions,
            #T = T,
        ).to(self.device)
        
        # Initialize continuous hypernetwork
        self.continuous_hypernetwork = ContinuousDirectionHyperNetworkCorrector(
            direction_dim = self.dim,
            target_param_size = total_param_size,
            hidden_dims = self.hidden_dims
        ).to(self.device)
        
        # Load the saved states
        self.direction_interpolation.load_state_dict(interpolation_state)
        self.continuous_hypernetwork.load_state_dict(hypernetwork_state)

        # Load LoRA states if available
        if 'lora_states' in checkpoint:
            self.lora_states = checkpoint['lora_states']
        else:
            print("WARNING: No LoRA states found in checkpoint - you need to load them separately!")

        if 'fourier_features_B' in checkpoint:
            self.continuous_hypernetwork.B = checkpoint['fourier_features_B'].to(self.device)
        
        print(f"Successfully loaded hypernetwork with:")
        print(f"  - {num_loras} LoRA directions")
        print(f"  - {total_param_size} total parameters")
        print(f"  - Attention mechanism and hypernetwork states restored")
        
        return True