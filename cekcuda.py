import torch

# Verify CUDA availability and device
print("CUDA Available:", torch.cuda.is_available())
print("Number Of GPUs:", torch.cuda.device_count())
print("PyTorch Version:", torch.__version__)

print(torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
