import joblib
from sklearn.tree import export_text

import numpy as np
import sounddevice as sd
import time

# Load trained model
model = joblib.load("sound_model.pkl")

# Recording settings
fs = 16000       # sample rate
duration = 2     # seconds per recording segment

print("Real-time sound monitoring started. Press Ctrl+C to stop.")

# print decision rules
rules = export_text(model, feature_names=['rms','peak','variance','dBFS'])
print("Rules: \n" + rules)

try:
    while True:
        # Record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        # Feature extraction
        rms = np.sqrt(np.mean(recording**2))
        peak = np.max(np.abs(recording))
        variance = np.var(recording)
        dBFS = 20 * np.log10(peak) if peak != 0 else -100

        features = np.array([[rms, peak, variance, dBFS]])

        # Make prediction
        label = model.predict(features)[0]

        # Print result
        print(f"RMS: {rms:.4f}, Peak: {peak:.4f}, Var: {variance:.6f}, dBFS: {dBFS:.2f} --> Predicted: {label}")

        # Small delay to avoid overlapping recordings
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")
