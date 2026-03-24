# INF2009 Edge Computing & Analytics Project by Group 24
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
* **Build System:** Flask, MQTT


## Contributors of Group 24
| Name                   | Student ID                                      |
| ---------------------- | ----------------------------------------------- |
| Chua Xin Jing          | [2302123](mailto:2302123@sit.singaporetech.edu.sg) |
| Liew DaiXuan           | [2302089](mailto:2302089@sit.singaporetech.edu.sg) |
| Shaw Aradhana          | [2302229](mailto:2302229@sit.singaporetech.edu.sg) |
| Xavier Teh Jun Ying    | [2301801](mailto:2301801@sit.singaporetech.edu.sg) |
| Yen Cheng Keh Yolanda  | [2302026](mailto:2302026@sit.singaporetech.edu.sg) |


## SetUp - (!! TO BE DONE IN YOUR RASPBERRY PI VIA REALVNC !!)
1. **Create virtual environment (unless your pi allows you to do step 2 directly then skip this step)**:
   Create a `.env` file in the root directory of the project folder.
   ```bash
   python -m venv env
   source env/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   bash download_models.sh
   ```

3. **Run app.py**:
   ```bash
   python app.py
   ```

4. **Copy the url into your laptop browser (raspberry-pi-ip will be the IP address you have set)**: 
    ```bash
   http://<raspberry-pi-ip>:5000/
   ```