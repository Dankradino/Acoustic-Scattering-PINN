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
            raise ValueError("LoRA rank r must be greater than 0.")

        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)  # Initialize B to zero for stability
        nn.init.zeros_(self.bias)

        self.weight.requires_grad = False

    def forward(self, x):
        if self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.alpha
            w_eff = self.weight + delta_w
        else:
            w_eff = self.weight

        return F.linear(x, w_eff, self.bias)


class FrozenActivationWrapper(nn.Module):
    """
    Wrapper that freezes activation parameters during LoRA training.
    
    CHANGE: Renamed from LoRAActivationWrapper to better reflect purpose.
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


class LoRAModelAdapter:
    """
    Unified adapter that converts any model to use LoRA.
    
    """
    
    @staticmethod
    def adapt_model(model: nn.Module,
                   r: int = 4,
                   alpha: float = 1.0,
                   freeze_base: bool = True,
                   freeze_activations: bool = True,
                   target_modules: Optional[List[str]] = None,
                   exclude_modules: Optional[List[str]] = None) -> nn.Module:
        """
        Convert any model to use LoRA in a single unified function.
        
        Args:
            model: Any PyTorch model (PINN, PINN_RFF, or custom)
            r: LoRA rank
            alpha: LoRA scaling factor
            freeze_base: Whether to freeze base model parameters
            freeze_activations: Whether to freeze activation function parameters
            target_modules: Optional list of module names to target
            exclude_modules: Optional list of module names to exclude
            
        Returns:
            LoRA-adapted model (preserves all original model attributes)
        """
        # Create a deep copy to preserve original model
        adapted_model = copy.deepcopy(model)
        
        # Step 1: Replace linear layers with LoRA layers
        LoRAModelAdapter._replace_linear_layers(
            adapted_model, r, alpha, target_modules, exclude_modules
        )
        
        # Step 2: Freeze activation parameters if requested
        if freeze_activations:
            LoRAModelAdapter._freeze_activations(adapted_model)
        
        # Step 3: Freeze base parameters if requested
        if freeze_base:
            LoRAModelAdapter._freeze_base_parameters(adapted_model)
        
        return adapted_model
    
    @staticmethod
    def _replace_linear_layers(module: nn.Module,
                              r: int,
                              alpha: float,
                              target_modules: Optional[List[str]],
                              exclude_modules: Optional[List[str]]):
        """
        Recursively replace Linear layers with LoRALinear.
        """
        if target_modules is None:
            target_modules = []
        if exclude_modules is None:
            exclude_modules = []
        
        def should_replace(name: str) -> bool:
            if target_modules and not any(target in name for target in target_modules):
                return False
            if any(exclude in name for exclude in exclude_modules):
                return False
            return True
        
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and should_replace(name):
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
                # Recursively process children
                LoRAModelAdapter._replace_linear_layers(
                    child, r, alpha, target_modules, exclude_modules
                )
    
    @staticmethod
    def _freeze_activations(module: nn.Module):
        """
        Freeze all activation function parameters.
        """
        for name, child in module.named_children():
            # Check if it's an activation layer with parameters
            if isinstance(child, (nn.Tanh, nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid)):
                for param in child.parameters():
                    param.requires_grad = False
            elif hasattr(child, 'parameters'):
                # Recursively check custom activations
                LoRAModelAdapter._freeze_activations(child)
    
    @staticmethod
    def _freeze_base_parameters(model: nn.Module):
        """
        Freeze all parameters except LoRA parameters.
        """
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False



class LoRACheckpointManager:
    """
    Handles saving and loading of LoRA checkpoints.
    """
    
    @staticmethod
    def save_lora_weights(model: nn.Module, path: str, save_format: str = 'nested'):
        """
        Save LoRA weights in a format compatible with existing checkpoints.
        
        Args:
            model: LoRA-adapted model
            path: Save path
            save_format: 'nested' (old format) or 'flat' (new format)
        """
        lora_state = {}
        
        if save_format == 'nested':
            # OLD FORMAT: For compatibility with existing checkpoints
            # Saves as {module_name: {lora_A: ..., lora_B: ..., alpha: ...}}
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear) and module.r > 0:
                    lora_state[name] = {
                        'lora_A': module.lora_A.detach().cpu(),
                        'lora_B': module.lora_B.detach().cpu(),
                        'alpha': module.alpha,
                    }
        else:
            # NEW FORMAT: Flatter structure, easier to work with
            # Saves as {module_name.lora_A: ..., module_name.lora_B: ...}
            for name, param in model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    lora_state[name] = param.detach().cpu()
        
        torch.save(lora_state, path)
        print(f"Saved LoRA weights to {path} (format: {save_format})")
    
    @staticmethod
    def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
        """
        Load LoRA weights with automatic format detection.
        
        Args:
            model: LoRA-adapted model
            path: Path to checkpoint
            strict: Whether to enforce strict loading
        """
        checkpoint = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        
        # Auto-detect format
        if LoRACheckpointManager._is_nested_format(checkpoint):
            print(f"Loading LoRA weights from {path} (detected: nested format)")
            LoRACheckpointManager._load_nested_format(model, checkpoint)
        else:
            print(f"Loading LoRA weights from {path} (detected: flat format)")
            LoRACheckpointManager._load_flat_format(model, checkpoint, strict)
    
    @staticmethod
    def _is_nested_format(checkpoint: dict) -> bool:
        """Check if checkpoint is in nested format."""
        if not checkpoint:
            return False
        first_value = next(iter(checkpoint.values()))
        return isinstance(first_value, dict) and 'lora_A' in first_value
    
    @staticmethod
    def _load_nested_format(model: nn.Module, checkpoint: dict):
        """Load nested format: {module_name: {lora_A: ..., lora_B: ...}}"""
        loaded_count = 0
        for name, module in model.named_modules():
            if name in checkpoint and isinstance(module, LoRALinear):
                state = checkpoint[name]
                module.lora_A.data.copy_(state['lora_A'].to(module.lora_A.device))
                module.lora_B.data.copy_(state['lora_B'].to(module.lora_B.device))
                if 'alpha' in state:
                    module.alpha = state['alpha']
                loaded_count += 1
        print(f"Loaded {loaded_count} LoRA modules")
    
    @staticmethod
    def _load_flat_format(model: nn.Module, checkpoint: dict, strict: bool):
        """Load flat format: {module_name.lora_A: ..., module_name.lora_B: ...}"""
        model_dict = model.state_dict()
        lora_weights = {}
        
        for name, param in checkpoint.items():
            if name in model_dict and ('lora_A' in name or 'lora_B' in name):
                lora_weights[name] = param
        
        model.load_state_dict(lora_weights, strict=False)
        print(f"Loaded {len(lora_weights)} LoRA parameters")




class LoRAUtils:
    """
    Utility functions for working with LoRA models.
    """
    
    @staticmethod
    def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
        """Get only LoRA parameters (A and B matrices)."""
        return [param for name, param in model.named_parameters()
                if 'lora_A' in name or 'lora_B' in name]
    
    @staticmethod
    def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
        """Get statistics about model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'lora_A' in name or 'lora_B' in name)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lora_parameters': lora_params,
            'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
            'lora_percentage': 100 * lora_params / total_params if total_params > 0 else 0,
        }
    
    @staticmethod
    def print_parameter_stats(model: nn.Module):
        """Print parameter statistics in a readable format."""
        stats = LoRAUtils.get_parameter_stats(model)
        print("\n" + "="*60)
        print("LoRA Model Statistics")
        print("="*60)
        print(f"Total parameters:      {stats['total_parameters']:,}")
        print(f"Trainable parameters:  {stats['trainable_parameters']:,}")
        print(f"LoRA parameters:       {stats['lora_parameters']:,}")
        print(f"Trainable percentage:  {stats['trainable_percentage']:.2f}%")
        print(f"LoRA percentage:       {stats['lora_percentage']:.2f}%")
        print("="*60 + "\n")
    
    @staticmethod
    def create_optimizer(model: nn.Module,
                        optimizer_type: str = 'lbfgs',
                        lr: float = 1.0,
                        **kwargs) -> torch.optim.Optimizer:
        """
        Create optimizer for LoRA parameters.
        """
        lora_params = LoRAUtils.get_lora_parameters(model)
        
        if not lora_params:
            raise ValueError("No LoRA parameters found in model!")
        
        if optimizer_type.lower() == 'lbfgs':
            return LBFGS(
                lora_params,
                lr=lr,
                max_iter=kwargs.get('max_iter', 20),
                history_size=kwargs.get('history_size', 50),
                line_search_fn=kwargs.get('line_search_fn', 'strong_wolfe')
            )
        elif optimizer_type.lower() == 'adam':
            return torch.optim.Adam(lora_params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        



def create_lora_model(base_model: nn.Module,
                     r: int = 12,
                     alpha: float = 1.0,
                     freeze_base: bool = True,
                     freeze_activations: bool = True,
                     device: Optional[str] = None) -> nn.Module:
    """
    Main entry point: Convert any model to LoRA.
    
    REPLACES: init_with_Lora, init_with_Lora_rff, adapt_with_lora, 
              adapt_pinn_rff_with_lora
    
    Args:
        base_model: Any PyTorch model (PINN, PINN_RFF, custom)
        r: LoRA rank
        alpha: LoRA scaling factor
        freeze_base: Freeze base model parameters
        freeze_activations: Freeze activation parameters
        device: Device to move model to
    
    Returns:
        LoRA-adapted model ready for training
    
    Example:
        >>> reference_model = load_my_model()
        >>> lora_model = create_lora_model(reference_model, r=12, alpha=1.0)
        >>> optimizer = LoRAUtils.create_optimizer(lora_model, 'lbfgs', lr=1.0)
    """
    # Adapt the model
    lora_model = LoRAModelAdapter.adapt_model(
        base_model,
        r=r,
        alpha=alpha,
        freeze_base=freeze_base,
        freeze_activations=freeze_activations
    )
    
    # Move to device if specified
    if device is not None:
        lora_model = lora_model.to(device)
    elif hasattr(base_model, 'device'):
        lora_model = lora_model.to(base_model.device)
    
    # Print stats
    LoRAUtils.print_parameter_stats(lora_model)
    
    return lora_model


def create_lora_model_with_optimizer(base_model: nn.Module,
                                    config: dict,
                                    r: int = 12,
                                    alpha: float = 1.0) -> tuple:
    """
    Create LoRA model and optimizer together (common use case).
    
    REPLACES: The tuple-returning init_with_Lora and init_with_Lora_rff
    
    Args:
        base_model: Base model to adapt
        config: Configuration dict with 'lr', 'max_iter', etc.
        r: LoRA rank
        alpha: LoRA scaling factor
    
    Returns:
        (lora_model, optimizer) tuple
    
    Example:
        >>> model, optimizer = create_lora_model_with_optimizer(
        ...     reference_model, config, r=12, alpha=1.0
        ... )
    """
    # Create LoRA model
    lora_model = create_lora_model(
        base_model,
        r=r,
        alpha=alpha,
        device=config.get('device', None)
    )
    
    # Create optimizer
    optimizer = LoRAUtils.create_optimizer(
        lora_model,
        optimizer_type='lbfgs',
        lr=config.get('lr', 1.0),
        max_iter=config.get('max_iter', 20),
        history_size=50,
        line_search_fn='strong_wolfe'
    )
    
    return lora_model, optimizer


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================
# These ensure your existing code continues to work without changes

def init_with_Lora(model, config, r=12, alpha=1., custom_activation=None):
    """Backward compatibility wrapper."""
    return create_lora_model_with_optimizer(model, config, r, alpha)

def init_with_Lora_rff(original_model, config, r=12, alpha=1.0, custom_activation=None):
    """Backward compatibility wrapper."""
    return create_lora_model_with_optimizer(original_model, config, r, alpha)

def save_lora_weights(model, path):
    """Backward compatibility wrapper."""
    LoRACheckpointManager.save_lora_weights(model, path, save_format='nested')

def load_lora_weights(model, path):
    """Backward compatibility wrapper."""
    LoRACheckpointManager.load_lora_weights(model, path)

# Keep LoRAAdapter as alias for compatibility
LoRAAdapter = LoRAModelAdapter