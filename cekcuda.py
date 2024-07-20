import torch

# Verify CUDA availability and device
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("PyTorch version:", torch.__version__)

print(torch.version.cuda)
if torch.cuda.is_available():
    