import sounddevice as sd
import numpy as np
import time
import csv
import os
from datetime import datetime
from scipy.io.wavfile import write
import joblib

fs = 16000
duration = 5
log_file = "noise_log.csv"

""" ==================================================================================
* RMS -> how loud the overall environment during the recording 
! Guide: 
0.001 RMS -> very quiet
0.005 RMS -> quiet
0.01 RMS -> normal speaking
0.02+ -> loud

* Peak -> loudest instant sound during the recording 
! range between -1.0 to +1.0
near to 1.0 will mean very loud sound/ mic clipping

* Variance -> how unstable/fluctuate the sound is
higher variance -> chaotic noise, many changes , multiple people talking
lower variance -> stable sound, quiet room, constant background 

================================================================================== """


# Create CSV with header if not exists
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "rms", "peak", "variance", "dBFS", "label"])

def compute_features(audio):
    # numpy rms formula
    rms = np.sqrt(np.mean(np.square(audio)))
    peak = np.max(np.abs(audio))
    variance = np.var(audio)
    # formula to calculate decibel from rms 
    dBFS = 20 * np.log10(rms) if rms > 0 else -np.inf
    return rms, peak, variance, dBFS

# due to the formula used for db, we get dBFS (decibels full scale) -> measurement for amplitude in digital audio
# ! dbfs is used for DIGITAL MEASUREMENT while db is used for PHYSICAL SOUND PRESSURE 
# represents levels relative to maximum possible digital signal (0 dBFS) b4 distortion 
# uses negative scale where 0 dBFS is the peak
def classify_noise(dBFS):
    # * -45 dBFS means the sound is 45 decibles quieter than the loudest possible sound 
    if dBFS < -45:
        return "quiet"
    elif -45 <= dBFS < -35:
        return "normal"
    else:
        return "noisy"

while True:
    print("Recording...")
    
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()

    recording = recording.flatten()

    rms, peak, variance, dBFS = compute_features(recording)
    label = input("Enter environment label (quiet/normal/noisy): ")

    # Safe timestamp (works on Windows & Pi)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"{timestamp} | dBFS: {dBFS:.2f} | Label: {label}")

    # Save features
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, rms, peak, variance, dBFS, label])

    # OPTIONAL: Save raw audio only if noisy (saves storage)
    if label == "noisy":
        write(f"raw_{timestamp}.wav", fs, recording)

    time.sleep(5)