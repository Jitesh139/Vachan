import sounddevice as sd

print("--- Audio Input Devices ---")
devices = sd.query_devices()
default_in = sd.default.device[0]
for idx, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        is_def = " (DEFAULT)" if idx == default_in else ""
        print(f"[{idx}] {d['name']} - Channels: {d['max_input_channels']}{is_def}")
