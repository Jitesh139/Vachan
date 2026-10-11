import sys

print("Python version:", sys.version)

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available in torch:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
        print("Device capability:", torch.cuda.get_device_capability(0))
        print("CUDA version:", torch.version.cuda)
except ImportError:
    print("PyTorch not installed in this environment.")

try:
    import ctranslate2
    print("CTranslate2 version:", ctranslate2.__version__)
    cuda_types = ctranslate2.get_supported_compute_types("cuda")
    print("CTranslate2 CUDA supported compute types:", cuda_types)
except Exception as e:
    print("CTranslate2 CUDA check:", e)

try:
    import faster_whisper
    print("faster_whisper version:", faster_whisper.__version__)
except ImportError:
    print("faster_whisper not installed.")
