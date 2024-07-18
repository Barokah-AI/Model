import torch

# Verify CUDA availability and device
print("CUDA availables:", torch.cuda.is_available())
print("Number of GPUss:", torch.cuda.device_count())
print("PyTorch versions:", torch.__version__)

print(torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA device names:", torch.cuda.get_device_name(0))
