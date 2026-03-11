import joblib
from sklearn.tree import export_text

import numpy as np
import sounddevice as sd
import time
import os

# Get the absolute path to the repo root (current file's parent folders)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
MODEL_PATH = os.path.join(REPO_ROOT, "microphone", "sound_model.pkl")

try:
	# Load trained model
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: sound_model.pkl not found at {MODEL_PATH}")
    model = None
    
# Recording settings
fs = 16000  # sample rate
duration = 2  # seconds per recording segment

# to check for sound sensor not detected
_STALE_TIMEOUT = 10  # 10s no data -> prob not connected
_last_check_time = 0


def get_sound():
    global _last_check_time

    try:
        # Record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        # Check if recording is valid
        if recording is None or not np.any(recording):
            elapsed = time.time() - _last_check_time
            if elapsed > _STALE_TIMEOUT:
                raise RuntimeError("Sound sensor not detected")
            else:
                return None  # ignore short glitches
        _last_check_time = time.time()

        # Feature extraction
        rms = np.sqrt(np.mean(recording**2))
        peak = np.max(np.abs(recording))
        variance = np.var(recording)
        dBFS = 20 * np.log10(peak) if peak != 0 else -100

        # Make prediction
        features = np.array([[rms, peak, variance, dBFS]])
        label = model.predict(features)[0]

        # # Print result
        # print(
        #     f"RMS: {rms:.4f}, Peak: {peak:.4f}, Var: {variance:.6f}, dBFS: {dBFS:.2f} --> Predicted: {label}"
        # )

        # Prepare result
        payload = {
            "rms": rms,
            "peak": peak,
            "variance": variance,
            "dBFS": dBFS,
            "label": label,
        }

        return payload

    except Exception as e:
        print(f"Error in get_sound(): {e}")
        return None
