#!/usr/bin/env python3
import torch
import sys

# Print PyTorch and MPS info
print("="*60)
print("DEVICE VERIFICATION")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print("="*60)

# Exit so you can verify before running training
sys.exit(0)