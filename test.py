import torch
print(torch.__version__)  # Versión de PyTorch
print(torch.version.cuda)  # Versión de CUDA

torch.backends.cuda.matmul.allow_tf32 = True
print(torch.backends.cuda.matmul.allow_tf32)  # En teoria era para usar FlashAttention, sin embargo no logré hacerlo funcionar
