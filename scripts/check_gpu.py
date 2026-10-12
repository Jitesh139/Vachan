"""Report whether faster-whisper can use the NVIDIA GPU. Never fails the launcher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    """Print GPU status in one or two lines."""
    from vachan.stt.transcriber import add_cuda_dll_dirs

    try:
        import ctranslate2

        add_cuda_dll_dirs()
        count = ctranslate2.get_cuda_device_count()
        if count:
            types = ", ".join(sorted(ctranslate2.get_supported_compute_types("cuda")))
            print(f"[GPU] {count} CUDA device(s) usable by CTranslate2 {ctranslate2.__version__} ({types})")
            print("      Whisper large-v3-turbo will run on the GPU.")
        else:
            print("[GPU] No CUDA device - Whisper will run on the CPU (slower, model 'small').")
    except Exception as e:
        print(f"[GPU] CUDA not usable ({e}) - Whisper will run on the CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
