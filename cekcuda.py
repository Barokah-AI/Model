import torch

# Verify CUDA availability and device
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("PyTorch version:", torch.__version__)

# Mencetak versi CUDA yang tersedia di PyTorch
print(torch.version.cuda)

# Mengecek apakah CUDA tersedia di perangkat
if torch.cuda.is_available():
    # Mencetak nama perangkat CUDA (GPU) yang sedang digunakan
    print("CUDA device name:", torch.cuda.get_device_name(0))