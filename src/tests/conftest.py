import os
import torch

# Prevent PyTorch/OpenMP CPU thread access violation crashes on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
