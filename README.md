## Hand tracking

### 1. File Description
- `palm_detection_without_custom_op.tflite` (Palm detection) Model file: Downloaded from the [*mediapipe-models*] repository.
- `hand_landmark.tflite` (Landmark detection) Model file: Downloaded from the [*mediapipe*] repository.    
- `anchors.csv`,`hand_tracker.py` files：Downloaded from the [*hand_tracking*] repository.

### 2. Setup
```sh
pip install opencv-python tensorflow
```

### 3. Implementation
- ```python run.py --3d True``` for 3D Hand Tracking.
- ```python run.py --3d False``` for just Palm Tracking.

### 4. Results
#### Palm tracking: 
![Result](/res.gif?raw=true "Result: Palm Tracking")

### 5. Acknowledgements
- Thanks to @metalwhale for the python implementation of the mediapipe models.
- Thanks to mediapipe for opensourcing these models.

[*mediapipe-models*]: https://github.com/junhwanjang/mediapipe-models/tree/master/palm_detection/mediapipe_models
[*mediapipe*]: https://github.com/google/mediapipe/tree/master/mediapipe/models
[*hand_tracking*]: https://github.com/wolterlw/hand_tracking
