import torch

print("="*40)
print("PyTorch   :", torch.__version__)
print("CUDA avail:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU       :", torch.cuda.get_device_name(0))
    print("VRAM      :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
    x = torch.randn(100,100).cuda()
    print("GPU test  : ✅ OK")
else:
    print("GPU       : ❌ Không detect được")
    print("CUDA ver  :", torch.version.cuda)
print("="*40)