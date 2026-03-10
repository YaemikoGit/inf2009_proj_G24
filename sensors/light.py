import RPi.GPIO as GPIO
import time

SENSOR_PIN = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

_last_value = None
_last_change_time = time.time()
_STALE_TIMEOUT = 10  # seconds without change = likely unplugged

def get_light():
    global _last_value, _last_change_time
    try:
        value = GPIO.input(SENSOR_PIN)

        # If value changed, update tracking
        if value != _last_value:
            _last_value = value
            _last_change_time = time.time()

        # If pin hasn't changed in 10 seconds, likely unplugged
        elapsed = time.time() - _last_change_time
        if elapsed > _STALE_TIMEOUT:
            raise RuntimeError("Light sensor not detected")

        return not value

    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Light sensor not detected")
