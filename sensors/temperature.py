import time
import board
import adafruit_dht

dht = adafruit_dht.DHT11(board.D12)

def get_temperature():
    # sensor logic here
    try:
        temp = dht.temperature
        humidity = dht.humidity
        if temp is None or humidity is None:
            # Sometimes the sensor returns None
            raise RuntimeError("Failed to read from DHT11 sensor")
        return {'temperature': temp, 'humidity': humidity}
    except RuntimeError as e:
        # Could be a transient read error; re-raise as RuntimeError
        raise RuntimeError(f"Temperature sensor read failed: {e}")
