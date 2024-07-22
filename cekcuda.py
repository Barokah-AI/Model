# # Cek apakah CUDA tersedia dan informasi perangkat CUDA yang digunakan

# import torch

# # Verify CUDA availability and device
# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("PyTorch version:", torch.__version__)

# # Print CUDA version and device name
# print(torch.version.cuda)
# if torch.cuda.is_available():
#     print("CUDA device name:", torch.cuda.get_device_name(0))
