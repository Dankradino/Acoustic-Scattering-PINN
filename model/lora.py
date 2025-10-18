import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np
from typing import Union, List, Optional, Dict, Any
import copy

"""
This module implement LoRA adaptation related function for a reference PINN.
"""

class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models
    W_eff = W + alpha * (A @ B)
    The bias is from the original network.
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.lora_A = None
            self.lora_B = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.lora_A is not None:
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if self.lora_B is not None:
            nn.init.zeros_(self.lora_B)  # Initialize B to zero for stability
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.weight.requires_grad = False

    def forward(self, x):
        if self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.alpha
            w_eff = self.weight + delta_w
        else:
            w_eff = self.weight

        return F.linear(x, w_eff, self.bias)


class LoRAActivationWrapper(nn.Module):
    """
    Wrapper for activation functions with frozen parameters
    Keeps activation parameters fixed during LoRA training
    """
    def __init__(self, activation_fn):
        super().__init__()
        if isinstance(activation_fn, type):
            self.activation = activation_fn()
        else:
            self.activation = copy.deepcopy(activation_fn)
        
        # Freeze all activation parameters
        for param in self.activation.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.activation(x)


class LoRAAdapter:
    """
    General LoRA adapter that can convert any Siren-based model to use LoRA
    Enhanced to support custom activation functions
    """
    
    @staticmethod
    def replace_linear_with_lora(model: nn.Module, 
                                r: int = 4, 
                                alpha: float = 1.0, 
                                target_modules: Optional[List[str]] = None,
                                exclude_modules: Optional[List[str]] = None,
                                custom_activation: Optional[nn.Module] = None) -> nn.Module:
        """
        Replace all nn.Linear layers in a model with LoRALinear layers
        
        Args:
            model: The model to adapt
            r: LoRA rank
            alpha: LoRA scaling factor
            target_modules: List of module names to target (if None, targets all linear layers)
            exclude_modules: List of module names to exclude from LoRA adaptation
            custom_activation: Custom activation function to use (e.g., MyCustomActivation())
        
        Returns:
            Modified model with LoRA layers
        """
        if target_modules is None:
            target_modules = []
        if exclude_modules is None:
            exclude_modules = []
        
        def should_replace(name: str) -> bool:
            # If target_modules is specified, only replace those
            if target_modules and not any(target in name for target in target_modules):
                return False
            # Don't replace if in exclude list
            if any(exclude in name for exclude in exclude_modules):
                return False
            return True
        
        # Create a copy of the model to avoid modifying the original
        adapted_model = copy.deepcopy(model)
        
        # Replace linear layers recursively
        LoRAAdapter._replace_linear_recursive(adapted_model, r, alpha, should_replace)
        
        # Replace activation functions if specified
        if custom_activation is not None:
            LoRAAdapter._replace_activations_recursive(adapted_model, custom_activation)
        #print(adapted_model)
        return adapted_model
    
    @staticmethod
    def _replace_linear_recursive(module: nn.Module, r: int, alpha: float, should_replace_fn):
        """
        Recursively replace linear layers in a module
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and should_replace_fn(name):
                # Create LoRA replacement
                lora_layer = LoRALinear(
                    child.in_features, 
                    child.out_features, 
                    r=r, 
                    alpha=alpha, 
                    bias=child.bias is not None
                )
                
                # Copy weights from original layer
                with torch.no_grad():
                    lora_layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        lora_layer.bias.copy_(child.bias)
                
                # Replace the layer
                setattr(module, name, lora_layer)
            else:
                # Recursively process child modules
                LoRAAdapter._replace_linear_recursive(child, r, alpha, should_replace_fn)
    
    @staticmethod
    def _replace_activations_recursive(module: nn.Module, custom_activation: nn.Module):
        """
        Recursively replace activation functions in a module
        """
        for name, child in module.named_children():
            if isinstance(child, (nn.Tanh, nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid)):
                # Create new instance of custom activation
                if isinstance(custom_activation, type):
                    new_activation = LoRAActivationWrapper(custom_activation())
                else:
                    new_activation = LoRAActivationWrapper(copy.deepcopy(custom_activation))
                
                setattr(module, name, new_activation)
            else:
                # Recursively process child modules
                LoRAAdapter._replace_activations_recursive(child, custom_activation)
    
    @staticmethod
    def freeze_base_parameters(model: nn.Module):
        """
        Freeze all parameters except LoRA parameters
        Activation parameters are kept frozen as per requirements
        """
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    @staticmethod
    def unfreeze_all_parameters(model: nn.Module):
        """
        Unfreeze all parameters
        """
        for param in model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def get_lora_parameters(model: nn.Module):
        """
        Get only LoRA parameters for optimizer (excluding activation parameters)
        """
        return [param for name, param in model.named_parameters() 
                if 'lora_A' in name or 'lora_B' in name]
    
    @staticmethod
    def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
        """
        Get statistics about model parameters
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for name, p in model.named_parameters() 
                         if 'lora_A' in name or 'lora_B' in name)
        activation_params = sum(p.numel() for name, p in model.named_parameters() 
                               if 'alpha' in name)
        frozen_activation_params = sum(p.numel() for name, p in model.named_parameters() 
                                     if 'alpha' in name and not p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lora_parameters': lora_params,
            'activation_parameters': activation_params,
            'frozen_activation_parameters': frozen_activation_params,
            'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
            'lora_percentage': 100 * lora_params / total_params if total_params > 0 else 0,
            'activation_percentage': 100 * activation_params / total_params if total_params > 0 else 0
        }
    
    @staticmethod
    def print_parameter_stats(model: nn.Module):
        """
        Print parameter statistics
        """
        stats = LoRAAdapter.get_parameter_stats(model)
        print(f"Total parameters: {stats['total_parameters']:,}")
        print(f"Trainable parameters: {stats['trainable_parameters']:,}")
        print(f"LoRA parameters: {stats['lora_parameters']:,}")
        print(f"Activation parameters (total): {stats['activation_parameters']:,}")
        print(f"Activation parameters (frozen): {stats['frozen_activation_parameters']:,}")
        print(f"Trainable percentage: {stats['trainable_percentage']:.2f}%")
        print(f"LoRA percentage: {stats['lora_percentage']:.2f}%")
        print(f"Activation percentage: {stats['activation_percentage']:.2f}%")
    
    @staticmethod
    def save_lora_weights(model: nn.Module, path: str):
        """
        Save only LoRA weights (excluding activation weights)
        """
        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or ('bias' in name and param.requires_grad):
                lora_state[name] = param.data
        torch.save(lora_state, path)
    
    @staticmethod
    def load_lora_weights(model: nn.Module, path: str):
        """
        Load LoRA weights (excluding activation weights)
        """
        lora_state = torch.load(path)
        model_dict = model.state_dict()
        
        # Update only LoRA parameters
        for name, param in lora_state.items():
            if name in model_dict and ('lora_A' in name or 'lora_B' in name or 'bias' in name):
                model_dict[name].copy_(param)
        
        print(f"Loaded LoRA weights from {path}")
    
    @staticmethod
    def merge_lora_weights(model: nn.Module) -> nn.Module:
        """
        Merge LoRA weights into base weights and return a standard model
        """
        merged_model = copy.deepcopy(model)
        
        for name, module in merged_model.named_modules():
            if isinstance(module, LoRALinear):
                # Merge LoRA weights into base weights
                with torch.no_grad():
                    if module.r > 0:
                        delta_w = (module.lora_B @ module.lora_A) * module.alpha
                        merged_weight = module.weight + delta_w
                    else:
                        merged_weight = module.weight
                    
                    # Create a standard linear layer
                    merged_linear = nn.Linear(
                        module.in_features, 
                        module.out_features, 
                        bias=module.bias is not None
                    )
                    merged_linear.weight.copy_(merged_weight)
                    if module.bias is not None:
                        merged_linear.bias.copy_(module.bias)
                    
                    # Replace the LoRA layer with merged linear layer
                    # This requires finding the parent module and replacing the child
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent_module = merged_model
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, child_name, merged_linear)
                    else:
                        setattr(merged_model, child_name, merged_linear)
        
        return merged_model


class ConditionalLoRA(nn.Module):
    """
    Conditional LoRA that can handle both RFF and standard inputs
    Uses the same LoRA parameters but different base weights for different input types
    """
    def __init__(self, 
                 in_features_std: int,      # Standard input features
                 in_features_rff: int,      # RFF input features (2 * num_fourier_features)
                 out_features: int, 
                 r: int = 4, 
                 alpha: float = 1.0, 
                 bias: bool = True):
        super().__init__()
        
        self.in_features_std = in_features_std
        self.in_features_rff = in_features_rff
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
        # Base weights for both paths
        self.weight_std = nn.Parameter(torch.zeros(out_features, in_features_std))
        self.weight_rff = nn.Parameter(torch.zeros(out_features, in_features_rff))
        
        # Shared LoRA parameters - we'll use the larger dimension
        max_in_features = max(in_features_std, in_features_rff)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, max_in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.lora_A = None
            self.lora_B = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_parameters()
        
        # Freeze base weights by default
        self.weight_std.requires_grad = False
        self.weight_rff.requires_grad = False
    
    def _init_parameters(self):
        nn.init.normal_(self.weight_std, mean=0.0, std=0.02)
        nn.init.normal_(self.weight_rff, mean=0.0, std=0.02)
        
        if self.r > 0:
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, use_rff=False):
        """
        Args:
            x: input tensor
            use_rff: whether input is RFF-processed or standard
        """
        if use_rff:
            return self._forward_rff(x)
        else:
            return self._forward_std(x)
    
    def _forward_std(self, x):
        base_weight = self.weight_std
        if self.r > 0:
            # Truncate or pad LoRA matrices to match input dimension
            lora_A_truncated = self.lora_A[:, :self.in_features_std]
            delta_w = (self.lora_B @ lora_A_truncated) * self.alpha
            w_eff = base_weight + delta_w
        else:
            w_eff = base_weight
        return F.linear(x, w_eff, self.bias)
    
    def _forward_rff(self, x):
        base_weight = self.weight_rff
        if self.r > 0:
            # Truncate or pad LoRA matrices to match input dimension
            lora_A_truncated = self.lora_A[:, :self.in_features_rff]
            delta_w = (self.lora_B @ lora_A_truncated) * self.alpha
            w_eff = base_weight + delta_w
        else:
            w_eff = base_weight
        return F.linear(x, w_eff, self.bias)


class PINN_rff_LoRA(nn.Module):
    """
    LoRA-adapted version of PINN_RFF that matches your exact network structure
    """
    def __init__(self, original_model, r=4, alpha=1.0, custom_activation=None):
        super().__init__()
        
        # Copy essential attributes from original model
        self.input_dim = original_model.input_dim
        self.hidden_dim = original_model.hidden_dim
        self.output_dim = original_model.output_dim
        self.num_layers = original_model.num_layers
        self.num_fourier_features = original_model.num_fourier_features
        self.sigma_fourier = original_model.sigma_fourier
        self.device = original_model.device
        
        # Copy the RFF matrix B
        self.register_buffer("B", original_model.B.clone())
        
        # Build the network following your exact structure
        layers = []
        
        # First layer: Fourier features (2 * num_fourier_features) to hidden
        orig_first_layer = original_model.net[0]  # First layer from original model
        first_layer = LoRALinear(
            2 * self.num_fourier_features, 
            self.hidden_dim,
            r=r, alpha=alpha,
            bias=(orig_first_layer.bias is not None)
        )
        
        # Copy weights from original model
        with torch.no_grad():
            first_layer.weight.copy_(orig_first_layer.weight)
            if orig_first_layer.bias is not None and first_layer.bias is not None:
                first_layer.bias.copy_(orig_first_layer.bias)
        
        layers.append(first_layer)
        
        # Add activation after first layer
        if custom_activation is not None:
            if isinstance(custom_activation, type):
                activation = LoRAActivationWrapper(custom_activation())
            else:
                # Copy the activation from the original model if it exists
                orig_activation = original_model.net[1]  # Should be the activation
                if hasattr(orig_activation, 'alpha'):
                    # Copy the learned alpha parameter but freeze it
                    new_activation = copy.deepcopy(orig_activation)
                    activation = LoRAActivationWrapper(new_activation)
                else:
                    activation = LoRAActivationWrapper(copy.deepcopy(custom_activation))
        else:
            print('Not good bro')
            activation = LoRAActivationWrapper(nn.Tanh())
        
        layers.append(activation)
        
        # Add hidden layers with alternating linear layers and activations
        for i in range(self.num_layers - 1):
            # Linear layer index in original model: 2 + i*2 (accounting for activations)
            orig_layer_idx = 2 + i * 2
            orig_layer = original_model.net[orig_layer_idx]
            
            hidden_layer = LoRALinear(
                self.hidden_dim, 
                self.hidden_dim,
                r=r, alpha=alpha,
                bias=(orig_layer.bias is not None)
            )
            
            # Copy weights from original model
            with torch.no_grad():
                hidden_layer.weight.copy_(orig_layer.weight)
                if orig_layer.bias is not None and hidden_layer.bias is not None:
                    hidden_layer.bias.copy_(orig_layer.bias)
            
            layers.append(hidden_layer)
            
            # Add activation after each hidden layer (reuse the same activation instance)
            layers.append(activation)
        
        # Final layer: hidden to output
        orig_final_layer = original_model.net[-1]  # Last layer
        final_layer = LoRALinear(
            self.hidden_dim,
            self.output_dim,
            r=r, alpha=alpha,
            bias=(orig_final_layer.bias is not None)
        )
        
        # Copy weights from original final layer
        with torch.no_grad():
            final_layer.weight.copy_(orig_final_layer.weight)
            if orig_final_layer.bias is not None and final_layer.bias is not None:
                final_layer.bias.copy_(orig_final_layer.bias)
        
        layers.append(final_layer)
        
        # Create the sequential model matching your structure
        self.net = nn.Sequential(*layers)

        #print(self.RFF)
    
    def forward(self, coords, diff=False):
        """
        Forward pass through the LoRA-adapted network
        """
        coords = coords.clone().detach().requires_grad_(True)
        
        # Apply RFF transformation using the original B matrix
        x = self.fourier_feature_mapping(coords)
        
        # Pass through the network
        output = self.net(x)
        
        if diff:
            return output, coords
        return output
    
    def fourier_feature_mapping(self, x):
        """Apply Fourier feature mapping using the original B matrix"""
        return torch.cat([torch.sin(2*np.pi*x @ self.B), torch.cos(2*np.pi*x @ self.B)], dim=-1)


# Convenience functions for easy usage
def adapt_with_lora(model: nn.Module, 
                   r: int = 4, 
                   alpha: float = 1.0,
                   target_modules: Optional[List[str]] = None,
                   exclude_modules: Optional[List[str]] = None,
                   freeze_base: bool = True,
                   custom_activation: Optional[nn.Module] = None) -> nn.Module:
    """
    One-stop function to adapt any Siren model with LoRA
    
    Args:
        model: The Siren model to adapt
        r: LoRA rank
        alpha: LoRA scaling factor
        target_modules: Specific modules to target (None for all linear layers)
        exclude_modules: Modules to exclude from LoRA adaptation
        freeze_base: Whether to freeze base parameters
        custom_activation: Custom activation function (e.g., MyCustomActivation())
    
    Returns:
        LoRA-adapted model
    """
    adapter = LoRAAdapter()
    lora_model = adapter.replace_linear_with_lora(
        model, r=r, alpha=alpha, 
        target_modules=target_modules, 
        exclude_modules=exclude_modules,
        custom_activation=custom_activation
    )
    
    if freeze_base:
        adapter.freeze_base_parameters(lora_model)
    
    return lora_model


def create_lora_optimizer(model: nn.Module, lr: float = 1., **kwargs) -> torch.optim.Optimizer:
    """
    Create optimizer for LoRA parameters only (activation parameters are frozen)
    """
    lora_params = LoRAAdapter.get_lora_parameters(model)
    return torch.optim.Adam(lora_params, lr=lr, **kwargs)


def init_with_Lora(model, config, r=12, alpha=1., custom_activation=None):
    """
    Initialize LoRA with custom activation support
    
    Args:
        model: Original model
        config: Training configuration
        r: LoRA rank
        alpha: LoRA scaling factor
        custom_activation: Custom activation function (e.g., MyCustomActivation())
    """
    # Adapt with LoRA and custom activation
    lora_model = adapt_with_lora(model, r=r, alpha=alpha, custom_activation=custom_activation)
    lora_model = lora_model.to(model.device)
    print("\nLoRA-adapted model:")
    LoRAAdapter.print_parameter_stats(lora_model)
    
    # Create optimizer for LoRA and activation parameters only
    lora_params = LoRAAdapter.get_lora_parameters(lora_model)
    optimizer = LBFGS(
        lora_params, 
        lr=config['lr'], 
        history_size=50,
        max_iter=config['max_iter'], 
        line_search_fn='strong_wolfe'

    )
    return lora_model, optimizer


def adapt_pinn_rff_with_lora(original_model, r=4, alpha=1.0, custom_activation=None):
    """
    Convert an existing PINN_RFF model to use LoRA with frozen custom activation
    
    Args:
        original_model: Your existing PINN_RFF model
        r: LoRA rank
        alpha: LoRA scaling factor
        custom_activation: Custom activation function (parameters will be frozen)
    
    Returns:
        LoRA-adapted model
    """
    lora_model = PINN_rff_LoRA(original_model, r=r, alpha=alpha, custom_activation=custom_activation)
    lora_model = lora_model.to(original_model.device)
    
    # Freeze base parameters (activation parameters are already frozen in LoRAActivationWrapper)
    LoRAAdapter.freeze_base_parameters(lora_model)
    
    return lora_model


def init_with_Lora_rff(original_model, config, r=12, alpha=1.0, custom_activation=None):
    """
    Initialize LoRA with existing RFF model and frozen custom activation
    
    Args:
        original_model: Your existing PINN_RFF model with RFF 
        config: Training configuration
        r: LoRA rank
        alpha: LoRA scaling factor
        custom_activation: Custom activation function (parameters will be frozen during training)
    
    Returns:
        lora_model: LoRA-adapted model
        optimizer: Optimizer for LoRA parameters only (activation params frozen)
    """
    # Adapt with LoRA while preserving existing RFF and using frozen custom activation
    lora_model = adapt_pinn_rff_with_lora(original_model, r=r, alpha=alpha, custom_activation=custom_activation)
    
    print("\nLoRA-adapted PINN_RFF model (with existing RFF and frozen custom activation):")
    LoRAAdapter.print_parameter_stats(lora_model)
    
    # Create optimizer for LoRA parameters only (activation parameters are frozen)
    lora_params = LoRAAdapter.get_lora_parameters(lora_model)
    
    # Using LBFGS optimizer
    optimizer = LBFGS(lora_params, lr = config['lr'], max_iter=config['max_iter'], line_search_fn='strong_wolfe')  
    #optimizer = Adam(lora_params, lr = config['lr'])
    return lora_model, optimizer



def load_pinn_rff_lora_checkpoint(model: 'PINN_rff_LoRA', 
                                 lora_checkpoint_path: str, 
                                 base_model_checkpoint_path: Optional[str] = None,
                                 strict: bool = True,
                                 device: Optional[str] = None) -> 'PINN_rff_LoRA':
    """
    Load LoRA weights into a PINN_rff_LoRA model from checkpoint files
    
    Args:
        model: The PINN_rff_LoRA model to load weights into
        lora_checkpoint_path: Path to the LoRA weights checkpoint
        base_model_checkpoint_path: Optional path to base model checkpoint
        strict: Whether to strictly enforce that checkpoint keys match model keys
        device: Device to load the model on
        
    Returns:
        PINN_rff_LoRA model with loaded weights
    """
    
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Load base model weights if provided
    if base_model_checkpoint_path is not None:
        print(f"Loading base model weights from: {base_model_checkpoint_path}")
        base_checkpoint = torch.load(base_model_checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in base_checkpoint:
            base_state_dict = base_checkpoint['model_state_dict']
        elif 'state_dict' in base_checkpoint:
            base_state_dict = base_checkpoint['state_dict']
        else:
            base_state_dict = base_checkpoint
        
        # Load base weights (non-LoRA parameters)
        model_dict = model.state_dict()
        base_weights = {}
        
        for name, param in base_state_dict.items():
            # Only load non-LoRA parameters
            if 'lora_A' not in name and 'lora_B' not in name:
                if name in model_dict:
                    base_weights[name] = param
                    
        model.load_state_dict(base_weights, strict=False)
        print(f"Loaded {len(base_weights)} base model parameters")
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {lora_checkpoint_path}")
    lora_checkpoint = torch.load(lora_checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'lora_state_dict' in lora_checkpoint:
        lora_state_dict = lora_checkpoint['lora_state_dict']
    elif 'model_state_dict' in lora_checkpoint:
        lora_state_dict = lora_checkpoint['model_state_dict']
    elif 'state_dict' in lora_checkpoint:
        lora_state_dict = lora_checkpoint['state_dict']
    else:
        lora_state_dict = lora_checkpoint
    
    # Filter and load LoRA parameters
    model_dict = model.state_dict()
    lora_weights = {}
    
    for name, param in lora_state_dict.items():
        if name in model_dict:
            # Load LoRA parameters and trainable biases
            if 'lora_A' in name or 'lora_B' in name or ('bias' in name and model_dict[name].requires_grad):
                lora_weights[name] = param
    
    if len(lora_weights) == 0:
        print("Warning: No LoRA parameters found in checkpoint!")
        return model
    
    # Load LoRA weights
    model.load_state_dict(lora_weights, strict=False)
    print(f"Loaded {len(lora_weights)} LoRA parameters")
    
    # Print loaded parameter names for verification
    print("Loaded LoRA parameters:")
    for name in sorted(lora_weights.keys()):
        print(f"  - {name}: {lora_weights[name].shape}")
    
    return model
