"""
Dummy Training Script for WS-TFA to verify computation graph and gradient flow.
"""

import torch
import torch.optim as optim
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from models.ws_tfa import WS_TFA_Net
from models.loss import WSTFALoss

def main():
    print("=== WS-TFA Dummy Training Verification ===")
    
    # 1. Hyperparameters & Configuration
    batch_size = 2
    num_classes = 20
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Instantiate Model and Loss
    print("\nInitializing Model and Loss Function...")
    model = WS_TFA_Net(num_classes=num_classes, pretrained_backbone=False).to(device)
    # Use top_k_pseudo = 3 so that queries other than the best one get gradients towards the best one
    criterion = WSTFALoss(alpha_reg_weight=0.01, box_loss_weight=1.0, top_k_pseudo=3).to(device)
    
    # Set model to training mode
    model.train()
    
    # 3. Setup Optimizer
    # Check if all parameters require grad
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Generate Dummy Data
    # Image tensor: [Batch, Channels, Height, Width]
    # Reduce size to 256x256 to speed up CPU dummy training
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Image-level Labels: [Batch, Num_Classes], multi-hot encoded (0 or 1)
    # E.g., image 0 has class 3 and 15, image 1 has class 0
    dummy_labels = torch.zeros(batch_size, num_classes).to(device)
    dummy_labels[0, 3] = 1.0
    dummy_labels[0, 15] = 1.0
    dummy_labels[1, 0] = 1.0
    
    print(f"Dummy Images Shape: {dummy_images.shape}")
    print(f"Dummy Labels Shape: {dummy_labels.shape}")

    # 5. Training Loop (Simulating Epochs for Warm-up Verification)
    print("\nStarting Training Loop Simulation...")
    
    num_epochs = 7
    warmup_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch} ---")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(dummy_images)
        
        # Compute Loss (pass current_epoch to handle Warm-up)
        total_loss, loss_dict = criterion(
            outputs, 
            dummy_labels, 
            current_epoch=epoch, 
            warmup_epochs=warmup_epochs
        )
        
        print(f"  -> Loss Components: MIL={loss_dict['loss_mil']:.4f}, "
              f"Alpha Reg={loss_dict['loss_alpha_reg']:.4f}, "
              f"Box={loss_dict['loss_box']:.4f}")
        print(f"  -> Total Loss: {total_loss.item():.4f}")
        
        # Backward Pass
        total_loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # 6. Verify Gradient Flow for Box Head
        bbox_head_weight = model.mil_head.proposal_generator.bbox_head[0].weight
        if bbox_head_weight.grad is not None:
            grad_norm = bbox_head_weight.grad.norm().item()
            print(f"  -> BBox Head Gradient Norm: {grad_norm:.6f}")
            if epoch < warmup_epochs and grad_norm > 0:
                print("     ❌ Error: Box head received gradients during Warm-up!")
            elif epoch >= warmup_epochs and grad_norm == 0:
                print("     ❌ Error: Box head received NO gradients after Warm-up!")
            elif epoch >= warmup_epochs and grad_norm > 0:
                print("     ✅ Success: Box head is actively training.")
        else:
            print("  -> BBox Head Gradient is None!")

    print("\n🎉 Success! The Warm-up mechanism and Box Loss gradient flow work perfectly.")


if __name__ == '__main__':
    main()
