import sounddevice as sd
import numpy as np
import csv
import os
import time
from datetime import datetime
from scipy.io.wavfile import write

# ==================== CONFIG ====================
fs = 16000                # Sampling rate
chunk_duration = 1.5     # Duration of each chunk in seconds
chunk_size = int(fs * chunk_duration)
log_file = "noise_log.csv"
save_wav = True           # Save WAV for each chunk (optional)
# =================================================

# Create CSV file with header if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "rms", "peak", "variance", "dBFS", "label"])

def compute_features(audio):
    rms = np.sqrt(np.mean(np.square(audio)))
    peak = np.max(np.abs(audio))
    variance = np.var(audio)
    dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
    return rms, peak, variance, dBFS

print("Starting data collection... Press Ctrl+C to stop.")
try:
    while True:
        # Record a short chunk
        recording = sd.rec(chunk_size, samplerate=fs, channels=1)
        sd.wait()
        recording = recording.flatten()

        # Compute features
        rms, peak, variance, dBFS = compute_features(recording)

        # Ask user to label this chunk
        label = input("Enter environment label (quiet/normal/noisy): ")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"{timestamp} | dBFS: {dBFS:.2f} | Label: {label}")

        # Save features to CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, rms, peak, variance, dBFS, label])

        # Optionally save WAV
        if label == "noisy":
            write(f"raw_{timestamp}.wav", fs, recording)

        # Short pause before next chunk
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Data collection stopped.")
