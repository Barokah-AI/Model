import torch

# Verify CUDA availability and device
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
