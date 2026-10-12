"""Failure handling: missing model offline, no GPU, microphone stall/unplug."""

import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from vachan.core.models import Transcript
from vachan.core.pipeline import PipelineEvents, PipelineWorker
from vachan.core.settings import Settings
from vachan.parser import ReferenceParser

LOAD_SCRIPT = """
import sys
from vachan.core.settings import Settings
from vachan.stt.transcriber import WhisperTranscriber, ModelUnavailable
t = WhisperTranscriber(Settings(device=sys.argv[1], model_cpu=sys.argv[2]))
try:
    t.load(lambda m: None)
    print("LOADED", t.device, "|", t.fallback_reason)
except ModelUnavailable as e:
    print("UNAVAILABLE", e)
"""


def run_load(tmp_path, device: str, model: str, offline: bool) -> str:
    env = {"HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path)} if offline else {}
    out = subprocess.run(
        [sys.executable, "-c", LOAD_SCRIPT, device, model],
        capture_output=True, text=True, timeout=300, env={**os.environ, **env}, cwd=".",
    )  # fmt: skip
    return out.stdout.strip().splitlines()[-1]


def test_missing_model_offline_gives_clear_message(tmp_path):
    line = run_load(tmp_path, "cpu", "tiny", offline=True)
    assert line.startswith("UNAVAILABLE")
    assert "not downloaded" in line and "internet" in line


@pytest.mark.skipif(bool(os.environ.get("CI")), reason="downloads the 480 MB small model")
def test_forced_cuda_without_gpu_falls_back_to_cpu(tmp_path):
    line = run_load(tmp_path, "cuda", "small", offline=False)
    assert line.startswith("LOADED cpu")
    try:
        import ctranslate2

        has_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_gpu = False
    if not has_gpu:
        assert "CUDA" in line or "GPU" in line  # reason is reported to the UI


class StallingAudio:
    """Delivers audio, then 'unplugs' (no blocks), then comes back on restart."""

    device_name, sample_rate = "USB Mixer", 16000

    def __init__(self):
        self.unplugged = False
        self.starts = 0

    def start(self):
        self.starts += 1
        if self.starts > 2:
            self.unplugged = False  # device came back

    def stop(self):
        pass

    def get_level(self):
        return 0.0

    def read_latest(self, seconds):
        return np.zeros(int(seconds * 16000), np.float32)

    def seconds_since_audio(self):
        return 99.0 if self.unplugged else 0.0


class QuietSTT:
    device, model_name = "cpu", "fake"

    def load(self, progress=None):
        pass

    def has_speech(self, audio):
        return False

    def transcribe(self, audio):
        return Transcript("", "en", 0.0)


def test_mic_unplug_reported_and_recovered():
    audio = StallingAudio()
    errors, statuses = [], []
    done = threading.Event()

    def status(msg):
        statuses.append(msg)
        if msg.startswith("Audio restored"):
            done.set()

    worker = PipelineWorker(
        audio, QuietSTT(), ReferenceParser(), None, Settings(step_sec=0.05),
        PipelineEvents(error=errors.append, status=status),
    )  # fmt: skip
    worker.start()
    time.sleep(0.3)
    audio.unplugged = True
    assert done.wait(10), statuses
    worker.stop(wait=2)
    assert any("unplugged" in e for e in errors)
