"""Profile engine chain: NVIDIA -> Intel iGPU -> CPU, without loading real models."""

import numpy as np
import pytest

from vachan.core.models import Transcript
from vachan.core.settings import Settings
from vachan.stt import profiles
from vachan.stt.transcriber import ModelUnavailable


class FakeEngine:
    def __init__(self, engine, model, device, fail=None):
        self.spec, self.fail = (engine, model, device), fail
        self.device, self.model_name, self.fallback_reason = "not loaded", model, ""

    def load(self, progress=None):
        if self.fail:
            raise self.fail
        self.device = {"cuda": "cuda", "GPU": "openvino-gpu"}.get(self.spec[2], "cpu")

    def unload(self):
        self.device = "not loaded"

    def transcribe(self, audio, language=None):
        return Transcript("hello", language or "en", 0.1)


@pytest.fixture
def machine(monkeypatch):
    """Configurable fake machine; returns the list of engines that were built."""
    state = {"nvidia": False, "ov": [], "fail": {}, "built": []}

    def build(settings, engine, model, device):
        state["built"].append((engine, model, device))
        return FakeEngine(engine, model, device, state["fail"].get(device))

    monkeypatch.setattr(profiles, "has_nvidia", lambda: state["nvidia"])
    monkeypatch.setattr(profiles, "openvino_devices", lambda: state["ov"])
    monkeypatch.setattr(profiles, "build", build)
    return state


def test_auto_picks_accurate_on_nvidia(machine):
    machine["nvidia"] = True
    t = profiles.ProfileTranscriber(Settings(profile="auto"))
    t.load()
    assert (t.profile, t.device, t.model_name) == ("accurate", "cuda", "large-v3-turbo")


def test_auto_without_nvidia_uses_intel_gpu(machine):
    machine["ov"] = ["CPU", "GPU"]
    t = profiles.ProfileTranscriber(Settings(profile="auto"))
    t.load()
    assert (t.profile, t.device, t.model_name) == ("balanced", "openvino-gpu", "small")
    assert "no NVIDIA GPU" in t.fallback_reason
    assert machine["built"] == [("ov", "small", "GPU")]  # cuda skipped without building


def test_falls_through_to_cpu_when_igpu_fails(machine):
    machine["ov"], machine["fail"] = ["GPU"], {"GPU": RuntimeError("driver")}
    t = profiles.ProfileTranscriber(Settings(profile="fast"))
    t.load()
    assert (t.device, t.model_name) == ("cpu", "base")
    assert "GPU failed: driver" in t.fallback_reason
    assert t.transcribe(np.zeros(16000, np.float32)).text == "hello"


def test_offline_stops_the_chain(machine):
    machine["nvidia"], machine["fail"] = True, {"cuda": ModelUnavailable("offline")}
    with pytest.raises(ModelUnavailable):
        profiles.ProfileTranscriber(Settings(profile="accurate")).load()
    assert len(machine["built"]) == 1


def test_unload_then_new_profile(machine):
    s = Settings(profile="fast")
    t = profiles.ProfileTranscriber(s)
    t.load()
    t.unload()
    assert t.device == "not loaded"
    s.profile = "accurate"
    t.load()
    assert (t.profile, t.model_name) == ("accurate", "medium")


def test_unknown_profile_falls_back_to_balanced():
    assert profiles.resolve("turbo") == "balanced"
    assert profiles.resolve("custom") == "custom"
