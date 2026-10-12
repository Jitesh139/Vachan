# PyInstaller one-folder build (64-bit). Run via build.bat.
# Whisper models are NOT bundled - they download on first run (~0.5-1.6 GB).
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [("data/books.json", "data"), ("data/verse_counts.json", "data"),
         ("data/hindi_numbers.json", "data"), ("assets/fonts", "assets/fonts")]
if Path("data/bibles/vachan_bibles.db").exists():
    datas.append(("data/bibles/vachan_bibles.db", "data/bibles"))
datas += collect_data_files("faster_whisper")  # Silero VAD model
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")

# CUDA runtime from nvidia-cublas-cu12 / nvidia-cudnn-cu12, if installed (GPU build)
for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
    try:
        spec = importlib.util.find_spec(pkg)
    except ModuleNotFoundError:
        spec = None
    for root in (spec.submodule_search_locations if spec else []):
        for dll in (Path(root) / "bin").glob("*.dll"):
            binaries.append((str(dll), pkg.replace(".", "/") + "/bin"))

a = Analysis(["main.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=["vachan.bible.builder"],
             excludes=["torch", "tkinter", "matplotlib", "PySide6.QtWebEngineCore"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Vachan", console=False,
          target_arch="x86_64")
coll = COLLECT(exe, a.binaries, a.datas, name="Vachan")
