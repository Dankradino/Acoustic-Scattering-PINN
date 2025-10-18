import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .activation import *
from .phisk_module import DirectionInterpolationModule
import os
import glob
from collections import OrderedDict
import re

"""
This modules is only here to test the efficiency of LoRA interpolation 
without any corrective hypernetwork
"""

from model.phisk_module import DirectionInterpolationModule
class LoRAForward:
    def __init__(self, base_network, config, T = None):
        # Architecture components
        self.base_network = base_network
        self.config = config
        self.device = config['device']
        self.T = T
        # New components for continuous direction control
        self.lora_states = {}
        self.lora_directions = []
        self.direction_interpolation = None
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
            
            print(f"  Direction: {direction.cpu().numpy()}")
        
        # Convert directions to tensor
        self.lora_directions = torch.stack(self.lora_directions).to(self.device)
        
        # Initialize interpolation mechanism
        self.direction_interpolation = DirectionInterpolationModule(
            direction_dim=2,
            num_loras=len(self.lora_states),
            hidden_dim=64,
            lora_directions=self.lora_directions
        ).to(self.device)
        
        # Get parameter shapes from first LoRA state
        first_lora = list(self.lora_states.values())[0]
        self.param_shapes = self._get_param_shapes_from_lora_state(first_lora)
        
        print("Debug: Parameter shapes found:")
        for key, shape in list(self.param_shapes.items())[:5]:
            print(f"  {key}: {shape}")
        if len(self.param_shapes) > 5:
            print(f"  ... and {len(self.param_shapes) - 5} more")
        


        print(f"Initialized interpolation mechanism for {len(self.lora_states)} LoRAs")

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
        Compute parameters for continuous direction using ensemble 
        """
        if not hasattr(self, 'direction_interpolation') or self.direction_interpolation is None:
            raise ValueError("LoRA ensemble not loaded. Call load_lora_ensemble() first.")
        
        target_direction = target_direction.to(self.device)
        
        # Get interpolation weights for LoRA ensemble
        interpolation_weights = self.direction_interpolation(target_direction)
        
        # Compute weighted combination of LoRA parameters
        ensemble_params = self._compute_lora_ensemble(interpolation_weights)
        
        # Combine ensemble and continuous refinement
        combined_params = {}
        all_param_keys = self._get_all_param_keys(ensemble_params)
        
        for param_key in all_param_keys:
            ensemble_tensor = self._get_tensor_from_nested_dict(ensemble_params, param_key)
            self._set_tensor_in_nested_dict(combined_params, param_key, ensemble_tensor)
    
        return combined_params

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
    
    def load_lora_checkpoint(self, checkpoint_path):
        """
        Load saved lora, interpolation mechanism, and related data from checkpoint
        
        Args:
            checkpoint_path: Path to the saved checkpoint file
        """
        print(f"Loading lora checkpoint from {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        print(checkpoint.keys())  # to see all keys, including your RFF matrices
        
        # Extract saved data
        self.lora_directions = checkpoint['lora_directions'].to(self.device)
        
        # Initialize the interpolation mechanism and lora with correct dimensions
        num_loras = len(self.lora_directions)
        
        # Initialize interpolation mechanism
        self.direction_interpolation = DirectionInterpolationModule(
            direction_dim=2,
            num_loras=num_loras,
            lora_directions=self.lora_directions
        ).to(self.device)

        # Load LoRA states if available
        if 'lora_states' in checkpoint:
            self.lora_states = checkpoint['lora_states']
        else:
            print("WARNING: No LoRA states found in checkpoint - you need to load them separately!")

        print(f"Successfully loaded  loras with:")
        print(f"  - {num_loras} LoRA directions")
        return True

class LoRAInterpolator(torch.nn.Module):
    """
    Simple wrapper that makes the interpolation callable like model(points, direction)
    """
    def __init__(self, base_network, lora_dir, config, T):
        super().__init__()
        
        # Initialize the lora_interpolator
        self.lora_interpolator = LoRAForward(
            base_network=base_network,
            config=config,
            T = T,
        )
        
        
        # Set to eval mode
        self.lora_interpolator.load_lora_ensemble(lora_dir)
        self.lora_interpolator.direction_interpolation.eval()
        
        self.device = self.lora_interpolator.device
        
    def forward(self, points, direction):
        """
        Forward pass: model(points, direction) -> predictions
        
        Args:
            points: [N, 2] tensor of evaluation points
            direction: [2] tensor or list - target direction
            
        Returns:
            [N, 2] tensor of predicted field (real, imaginary)
        """
        # Convert inputs to tensors if needed
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=self.device)
        else:
            points = points.to(self.device)
            
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32, device=self.device)
        else:
            direction = direction.to(self.device)
        
        # Normalize direction
        direction = direction / torch.norm(direction)
        
        # Get adapted parameters for this direction
        with torch.no_grad():
            pred_params = self.lora_interpolator.get_continuous_direction_params(direction)
            
            # Forward pass through adapted network
            output = self.lora_interpolator.forward_with_lora_params(points, pred_params)
        
        return output
    
def load_interpolated_model(base_network, lora_dir, config, T = None):
    """
    Convenience function to load a directional hypernetwork model
    
    Args:
        base_network: Your base SIREN network
        lora_dir: Directory containing LoRA weights  
        config: Your config dict
        
    Returns:
        DirectionalHyperNetwork that can be called as model(points, direction)
    """
    return LoRAInterpolator(base_network, lora_dir, config, T = T)
