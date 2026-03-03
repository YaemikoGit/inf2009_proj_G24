import sounddevice as sd
import numpy as np
import time
import csv
from datetime import datetime
from scipy.io.wavfile import write

# sampling rate (16 kHz), 16,000 samples per second.
fs = 16000 
# each recording lasts 5 seconds
duration = 5 
log_file = "noise_log.csv"

def compute_features(audio):
    # rms = root mean square (perceived loudness)
    rms = np.sqrt(np.mean(np.square(audio)))
    # peak = maximum absolute value (sudden spikes)
    peak = np.max(np.abs(audio))
    # how much the signal fluctuates (noise stability)
    variance = np.var(audio)
    # converts RMS to decibels (dB) - standard noise measurement 
    # -np.inf is to ensure no errors if rms is 0
    db = 20 * np.log10(rms) if rms > 0 else -np.inf
    return rms, peak, variance, db

while True:
    print("Recording...")
    # records audio for 5 seconds
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    # pause the script until recording ends 
    sd.wait()

    # converts from 2D to 1D -> easier to compute features 
    recording = recording.flatten()

    print(f"recording: {recording}")

    rms, peak, variance, db = compute_features(recording)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"{timestamp} | dB: {db:.2f}")

    # Save features to CSV
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, rms, peak, variance, db])

    # OPTIONAL: Save raw audio locally
    write(f"raw_{timestamp}.wav", fs, recording)

    time.sleep(5)