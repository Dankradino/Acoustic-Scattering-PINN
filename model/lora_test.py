"""
Test script to verify backward compatibility of simplified LoRA implementation.
This ensures existing checkpoints load correctly and produce identical results.
"""

import torch
import torch.nn as nn
import numpy as np
from lora import (
    create_lora_model,
    create_lora_model_with_optimizer,
    LoRACheckpointManager,
    LoRAUtils,
    LoRALinear,  # Added for isinstance checks
    # Backward compatibility imports
    init_with_Lora,
    save_lora_weights,
    load_lora_weights,
)


# ============================================================================
# Mock Models for Testing
# ============================================================================

class SimplePINN(nn.Module):
    """Simple PINN model for testing"""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class PINN_RFF(nn.Module):
    """PINN with Random Fourier Features"""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, 
                 num_layers=3, num_fourier_features=256, sigma_fourier=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_fourier_features = num_fourier_features
        self.sigma_fourier = sigma_fourier
        self.device = 'cpu'
        
        # RFF matrix B (this should be preserved!)
        B = torch.randn(input_dim, num_fourier_features) * sigma_fourier
        self.register_buffer("B", B)
        
        # Build network
        layers = []
        layers.append(nn.Linear(2 * num_fourier_features, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def fourier_feature_mapping(self, x):
        """Apply RFF transformation"""
        return torch.cat([
            torch.sin(2 * np.pi * x @ self.B),
            torch.cos(2 * np.pi * x @ self.B)
        ], dim=-1)
    
    def forward(self, x):
        x_rff = self.fourier_feature_mapping(x)
        return self.net(x_rff)


# ============================================================================
# Test Functions
# ============================================================================

def test_basic_adaptation():
    """Test 1: Basic model adaptation"""
    print("\n" + "="*60)
    print("TEST 1: Basic Model Adaptation")
    print("="*60)
    
    # Create base model
    base_model = SimplePINN(input_dim=2, hidden_dim=64, output_dim=1)
    print("✓ Created base model")
    
    # Adapt with LoRA
    lora_model = create_lora_model(base_model, r=12, alpha=1.0)
    print("✓ Created LoRA model")
    
    # Check statistics
    stats = LoRAUtils.get_parameter_stats(lora_model)
    assert stats['lora_parameters'] > 0, "No LoRA parameters found!"
    print(f"✓ LoRA parameters: {stats['lora_parameters']:,}")
    print(f"✓ Trainable percentage: {stats['trainable_percentage']:.2f}%")
    
    # Test forward pass
    test_input = torch.randn(10, 2)
    output = lora_model(test_input)
    assert output.shape == (10, 1), f"Wrong output shape: {output.shape}"
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    print("\n✅ TEST 1 PASSED")


def test_rff_model_adaptation():
    """Test 2: RFF model adaptation (preserve buffers)"""
    print("\n" + "="*60)
    print("TEST 2: RFF Model Adaptation (Buffer Preservation)")
    print("="*60)
    
    # Create RFF model
    base_model = PINN_RFF(input_dim=2, hidden_dim=64, num_fourier_features=256)
    original_B = base_model.B.clone()
    print("✓ Created RFF model")
    print(f"  Original B matrix shape: {original_B.shape}")
    
    # Adapt with LoRA
    lora_model = create_lora_model(base_model, r=12, alpha=1.0)
    print("✓ Created LoRA model")
    
    # Check B matrix preserved
    assert hasattr(lora_model, 'B'), "RFF B matrix not preserved!"
    assert torch.allclose(lora_model.B, original_B), "B matrix changed!"
    print("✓ RFF B matrix correctly preserved")
    
    # Test forward pass
    test_input = torch.randn(10, 2)
    output = lora_model(test_input)
    assert output.shape == (10, 1), f"Wrong output shape: {output.shape}"
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    print("\n✅ TEST 2 PASSED")


def test_checkpoint_compatibility_nested():
    """Test 3: Backward compatibility with nested checkpoint format"""
    print("\n" + "="*60)
    print("TEST 3: Nested Checkpoint Format (Old Format)")
    print("="*60)
    
    # Create base model
    base_model = SimplePINN()
    base_state = base_model.state_dict()  # Save for reuse
    
    # Create first LoRA model
    lora_model = create_lora_model(base_model, r=12, alpha=1.0)
    
    # Modify LoRA weights to non-zero values so we can verify they load correctly
    with torch.no_grad():
        for name, module in lora_model.named_modules():
            if isinstance(module, LoRALinear) and module.r > 0:
                module.lora_A.fill_(0.5)
                module.lora_B.fill_(0.3)
    
    # Save in nested format (old format)
    LoRACheckpointManager.save_lora_weights(
        lora_model, '/tmp/test_nested.pth', save_format='nested'
    )
    print("✓ Saved checkpoint in nested format")
    
    # Create new base model with SAME weights
    base_model2 = SimplePINN()
    base_model2.load_state_dict(base_state)
    
    # Create second LoRA model from same base
    lora_model2 = create_lora_model(base_model2, r=12, alpha=1.0)
    
    LoRACheckpointManager.load_lora_weights(lora_model2, '/tmp/test_nested.pth')
    print("✓ Loaded checkpoint")
    
    # Verify LoRA weights match
    for (name1, module1), (name2, module2) in zip(
        lora_model.named_modules(), lora_model2.named_modules()
    ):
        if isinstance(module1, LoRALinear) and isinstance(module2, LoRALinear):
            if module1.r > 0:
                assert torch.allclose(module1.lora_A, module2.lora_A, atol=1e-6), \
                    f"lora_A mismatch in {name1}"
                assert torch.allclose(module1.lora_B, module2.lora_B, atol=1e-6), \
                    f"lora_B mismatch in {name1}"
    print("✓ LoRA weights match")
    
    # Verify outputs match
    test_input = torch.randn(5, 2)
    output1 = lora_model(test_input)
    output2 = lora_model2(test_input)
    
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs don't match!"
    print("✓ Model outputs match after loading")
    
    print("\n✅ TEST 3 PASSED")


def test_checkpoint_compatibility_flat():
    """Test 4: New flat checkpoint format"""
    print("\n" + "="*60)
    print("TEST 4: Flat Checkpoint Format (New Format)")
    print("="*60)
    
    # Create base model
    base_model = SimplePINN()
    
    # Save base model state for reuse
    base_state = base_model.state_dict()
    
    # Create first LoRA model
    lora_model = create_lora_model(base_model, r=12, alpha=1.0)
    
    # Modify LoRA weights to non-zero values so we can verify they load correctly
    with torch.no_grad():
        for name, param in lora_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.fill_(0.7)
    
    # Save in flat format
    LoRACheckpointManager.save_lora_weights(
        lora_model, '/tmp/test_flat.pth', save_format='flat'
    )
    print("✓ Saved checkpoint in flat format")
    
    # Create new base model with SAME weights
    base_model2 = SimplePINN()
    base_model2.load_state_dict(base_state)
    
    # Create second LoRA model from same base
    lora_model2 = create_lora_model(base_model2, r=12, alpha=1.0)
    
    LoRACheckpointManager.load_lora_weights(lora_model2, '/tmp/test_flat.pth')
    print("✓ Loaded checkpoint")
    
    # Verify LoRA parameters match
    for (name1, param1), (name2, param2) in zip(
        lora_model.named_parameters(), lora_model2.named_parameters()
    ):
        if 'lora_A' in name1 or 'lora_B' in name1:
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Parameter mismatch: {name1}"
    print("✓ LoRA parameters match")
    
    # Verify outputs match
    test_input = torch.randn(5, 2)
    output1 = lora_model(test_input)
    output2 = lora_model2(test_input)
    
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs don't match!"
    print("✓ Model outputs match after loading")
    
    print("\n✅ TEST 4 PASSED")


def test_backward_compatible_api():
    """Test 5: Backward compatible function names"""
    print("\n" + "="*60)
    print("TEST 5: Backward Compatible API")
    print("="*60)
    
    # Create base model
    base_model = SimplePINN()
    base_state = base_model.state_dict()  # Save for reuse
    
    config = {'lr': 1.0, 'max_iter': 20, 'device': 'cpu'}
    
    # Use old API
    lora_model, optimizer = init_with_Lora(base_model, config, r=12, alpha=1.0)
    print("✓ init_with_Lora works")
    
    # Modify LoRA weights to known values
    with torch.no_grad():
        for name, param in lora_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.fill_(0.42)
    
    # Save using old function
    save_lora_weights(lora_model, '/tmp/test_old_api.pth')
    print("✓ save_lora_weights works")
    
    # Create new base model with SAME weights
    base_model2 = SimplePINN()
    base_model2.load_state_dict(base_state)
    
    # Load using old function
    lora_model2 = create_lora_model(base_model2, r=12, alpha=1.0)
    load_lora_weights(lora_model2, '/tmp/test_old_api.pth')
    print("✓ load_lora_weights works")
    
    # Verify LoRA parameters match
    for (name1, param1), (name2, param2) in zip(
        lora_model.named_parameters(), lora_model2.named_parameters()
    ):
        if 'lora_A' in name1 or 'lora_B' in name1:
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Parameter mismatch: {name1}"
    print("✓ LoRA parameters match")
    
    # Verify outputs match
    test_input = torch.randn(5, 2)
    output1 = lora_model(test_input)
    output2 = lora_model2(test_input)
    
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs don't match!"
    print("✓ Model outputs match")
    
    print("\n✅ TEST 5 PASSED")


def test_optimizer_creation():
    """Test 6: Optimizer creation"""
    print("\n" + "="*60)
    print("TEST 6: Optimizer Creation")
    print("="*60)
    
    base_model = SimplePINN()
    lora_model = create_lora_model(base_model, r=12, alpha=1.0)
    
    # Test LBFGS
    optimizer_lbfgs = LoRAUtils.create_optimizer(
        lora_model, 'lbfgs', lr=1.0, max_iter=20
    )
    assert optimizer_lbfgs is not None
    print("✓ LBFGS optimizer created")
    
    # Test Adam
    optimizer_adam = LoRAUtils.create_optimizer(
        lora_model, 'adam', lr=0.001
    )
    assert optimizer_adam is not None
    print("✓ Adam optimizer created")
    
    # Verify only LoRA params in optimizer
    lora_params = LoRAUtils.get_lora_parameters(lora_model)
    optimizer_param_ids = {id(p) for group in optimizer_lbfgs.param_groups for p in group['params']}
    lora_param_ids = {id(p) for p in lora_params}
    
    assert optimizer_param_ids == lora_param_ids, "Optimizer has wrong parameters!"
    print("✓ Optimizer contains only LoRA parameters")
    
    print("\n✅ TEST 6 PASSED")


def test_training_step():
    """Test 7: Simple training step"""
    print("\n" + "="*60)
    print("TEST 7: Training Step")
    print("="*60)
    
    # Create model and optimizer
    base_model = SimplePINN()
    config = {'lr': 1.0, 'max_iter': 1, 'device': 'cpu'}
    lora_model, optimizer = create_lora_model_with_optimizer(
        base_model, config, r=12, alpha=1.0
    )
    
    # Create dummy data
    x = torch.randn(100, 2)
    y = torch.randn(100, 1)
    
    # Get initial loss
    lora_model.train()
    output = lora_model(x)
    initial_loss = nn.MSELoss()(output, y)
    print(f"✓ Initial loss: {initial_loss.item():.6f}")
    
    # Training step
    def closure():
        optimizer.zero_grad()
        output = lora_model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    # Check loss changed
    output = lora_model(x)
    final_loss = nn.MSELoss()(output, y)
    print(f"✓ Final loss: {final_loss.item():.6f}")
    
    assert final_loss != initial_loss, "Loss didn't change!"
    print("✓ Training step successful")
    
    print("\n✅ TEST 7 PASSED")


def test_parameter_freezing():
    """Test 8: Parameter freezing"""
    print("\n" + "="*60)
    print("TEST 8: Parameter Freezing")
    print("="*60)
    
    base_model = SimplePINN()
    lora_model = create_lora_model(
        base_model, r=12, alpha=1.0, 
        freeze_base=True, freeze_activations=True
    )
    
    # Count trainable parameters
    base_params = []
    lora_params = []
    activation_params = []
    
    for name, param in lora_model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
            assert param.requires_grad, f"LoRA param {name} should be trainable!"
        elif 'weight' in name or ('bias' in name and not param.requires_grad):
            base_params.append(param)
            assert not param.requires_grad, f"Base param {name} should be frozen!"
    
    print(f"✓ Base parameters frozen: {len(base_params)}")
    print(f"✓ LoRA parameters trainable: {len(lora_params)}")
    
    assert len(lora_params) > 0, "No trainable LoRA parameters!"
    
    print("\n✅ TEST 8 PASSED")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING LORA SIMPLIFICATION COMPATIBILITY TESTS")
    print("="*60)
    
    tests = [
        test_basic_adaptation,
        test_rff_model_adaptation,
        test_checkpoint_compatibility_nested,
        test_checkpoint_compatibility_flat,
        test_backward_compatible_api,
        test_optimizer_creation,
        test_training_step,
        test_parameter_freezing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The simplified LoRA implementation is fully backward compatible.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)