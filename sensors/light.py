import RPi.GPIO as GPIO

SENSOR_PIN = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def get_light():
    try:
        # Read pin
        value = GPIO.input(SENSOR_PIN)
        # Convert to light/dark
        return not value
    except Exception:
        # Any unexpected GPIO error
        raise RuntimeError("Light sensor not detected")