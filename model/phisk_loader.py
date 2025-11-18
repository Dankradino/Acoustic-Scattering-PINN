import torch
from torch import nn
from .activation import *
from Trainer.phisk2D import PHISK_Trainer2D

"""
This module implements a loader for PHISK 
"""



class DirectionalHyperNetwork(nn.Module):
    """
    Simple wrapper that makes the hypernetwork callable like model(points, direction)
    """
    def __init__(self, reference_network, checkpoint_path, config, hconfig):
        super().__init__()
        
        # Initialize the trainer (this contains all the hypernetwork logic)
        self.trainer = PHISK_Trainer2D(
            reference_network=reference_network,
            hypernetwork_path=checkpoint_path,
            dataloader={},  # Empty for inference
            loss_fn=nn.MSELoss(),  # Dummy loss
            config=config,
            hconfig=hconfig
        )

        # Set to eval mode
        self.trainer.direction_interpolation.eval()
        self.trainer.continuous_hypernetwork.eval()
        
        self.device = self.trainer.device
        
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
            pred_params, _ = self.trainer.get_continuous_direction_params(direction)
            
            # Forward pass through adapted network
            output = self.trainer.forward_with_lora_params(points, pred_params)
        
        return output

def load_directional_model(reference_network, checkpoint_path, config, hconfig):
    """
    Convenience function to load a directional hypernetwork model
    
    Args:
        reference_network: Your base SIREN network
        lora_dir: Directory containing LoRA weights  
        checkpoint_path: Path to trained hypernetwork checkpoint
        config: Your config dict
        hconfig: Your hypernetwork config
        
    Returns:
        DirectionalHyperNetwork that can be called as model(points, direction)
    """
    return DirectionalHyperNetwork(reference_network, checkpoint_path, config, hconfig)