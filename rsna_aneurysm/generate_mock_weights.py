#!/usr/bin/env python3
"""Generate a mock weights file for testing."""
import torch
from model import Compact3DCNN

def main():
    # Initialize model with random weights
    model = Compact3DCNN(num_classes=14)
    
    # Create a state dict with random weights
    state_dict = {
        'state_dict': model.state_dict(),
        'epoch': 0,
        'best_score': 0.0,
    }
    
    # Save to file
    torch.save(state_dict, 'mock_weights.pth')
    print("Mock weights saved to mock_weights.pth")

if __name__ == "__main__":
    main()
