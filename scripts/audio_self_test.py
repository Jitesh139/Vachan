import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from vachan.audio.source import AudioDeviceManager, LiveAudioSource
from vachan.stt.transcriber import WhisperTranscriber


def run_self_test(device_index: int = 1, duration_sec: float = 10.0) -> None:
    print("=" * 60)
    print("AUDIO SELF-TEST: 10-Second Capture & Transcription")
    print("=" * 60)

    # 1. Query device
    devices = AudioDeviceManager.get_input_devices()
    chosen_dev = None
    for d in devices:
        if d["index"] == device_index:
            chosen_dev = d
            break

    print(f"Target Device Index: {device_index}")
    if chosen_dev:
        print(f"Device Name:         {chosen_dev['name']}")
        print(f"Channels:            {chosen_dev['channels']}")
        print(f"Default Sample Rate: {chosen_dev['default_samplerate']} Hz")
    else:
        print("Device index not found in input list; using default.")
        device_index = None

    # 2. Record 10 seconds
    source = LiveAudioSource(
        device_index=device_index,
        sample_rate=16000,
        chunk_duration_sec=duration_sec,
        input_gain=1.0,
        enable_normalization=True,
    )

    recorded_chunks = []

    def on_chunk(chunk: np.ndarray) -> None:
        recorded_chunks.append(chunk)

    print(f"\nRecording {duration_sec} seconds of audio now... (speak or silence)")
    source.start(on_chunk)

    start_time = time.time()
    while time.time() - start_time < duration_sec + 0.5:
        time.sleep(0.2)
        level = source.get_level()
        bars = int(level * 40)
        print(f"\rLevel Meter: [{'#' * bars}{'.' * (40 - bars)}] {level:.3f}", end="", flush=True)

    source.stop()
    print("\nRecording stopped.")

    if not recorded_chunks:
        print("ERROR: No audio chunks captured!")
        return

    audio = recorded_chunks[0]
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    print(f"\nCaptured Audio Stats:")
    print(f"  Shape:       {audio.shape}")
    print(f"  Dtype:       {audio.dtype}")
    print(f"  Sample Rate: {source.target_sample_rate} Hz (target)")
    print(f"  Device SR:   {source.active_sample_rate} Hz (stream)")
    print(f"  Channels:    {source.channels}")
    print(f"  RMS:         {rms:.5f}")
    print(f"  Peak:        {peak:.5f}")
    print(f"  Duration:    {len(audio) / 16000:.2f}s")

    # 3. Transcribe
    print("\nLoading Whisper transcriber...")
    transcriber = WhisperTranscriber(model_size="large-v3", device="auto")
    transcriber.load_model()
    print(f"Whisper running on: {transcriber.active_device.upper()} ({transcriber.active_compute_type})")

    print("\nTranscribing 10s audio...")
    t0 = time.time()
    text, lang = transcriber.transcribe_with_info(audio)
    t_elapsed = time.time() - t0

    print(f"\nTranscription Results ({t_elapsed:.2f}s):")
    print(f"  Detected Language: {lang}")
    print(f"  Transcript Text:   '{text}'")
    print("=" * 60)


if __name__ == "__main__":
    run_self_test(device_index=1, duration_sec=10.0)
