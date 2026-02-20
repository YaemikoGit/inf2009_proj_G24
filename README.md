# INF2009 Edge Computing & Analytics Project
## Smart classroom analytics system as a functional prototype that integrates vision-based student attention estimation together with classroom environmental sensing
This project proposes an edge-based classroom environment analytics system that integrates attention estimation with environmental sensing to provide meaningful insights into learning conditions. In this project, student attention is inferred through visual behavioural cues such as head orientation and gaze direction, which are widely used indicators of engagement in smart classroom research.

## System Architecture
* **Raspberry Pi 5:** primary edge processing unit.
* **Camera (WebCam):** used for computer vision-based student attention analysis; Processes video streams locally to estimate attention-related features (e.g., head orientation and gaze direction); Raw videos should not be transmitted off-device, ensuring privacy.
* **Noise Sensor (Microphone Module):** monitor ambient noise levels.
* **Humidity Sensors:** measure humidity to assess thermal comfort conditions.
* **Temperature Sensor:** capture classroom temperature to monitor thermal conditions.
* **Air Quality Sensor:** measure CO2 and other air quality indicators to evaluate indoor environmental conditions .


## Technologies Used
* **Language:** Python, JavaScript
* **UI:** HTML, BootStrap 
* **Backend & Database:** Neon Postgres (TBC)
* **Build System:** Flask

## Contributors

