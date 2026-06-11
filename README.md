# AI Object Detection Assistant

## Overview

This project is a real-time AI-powered object detection assistant built with Python, OpenCV, and Google's Text-to-Speech (gTTS) service.

The program uses a webcam to detect objects in its field of view using a pre-trained MobileNet SSD neural network. Detected objects are highlighted with bounding boxes and labels on the screen. Every five seconds, the program verbally announces the objects it has detected, making it useful for accessibility and computer vision demonstrations.

---

## Features

* Real-time webcam object detection
* Bounding boxes around detected objects
* Object labels with confidence scores
* Voice announcements using Google Text-to-Speech
* Full-screen display mode
* Multi-threaded speech processing to prevent video lag
* Automatic cleanup of temporary audio files

---

## Technologies Used

* Python
* OpenCV (Computer Vision)
* NumPy
* gTTS (Google Text-to-Speech)
* Pygame
* MobileNet SSD Deep Learning Model

---

## Required Python Modules

Install the required dependencies using:

```bash
pip install opencv-python numpy gtts pygame
```

### Imported Modules

```python
import cv2
import numpy as np
import time
import threading
import queue
from gtts import gTTS
import pygame
import os
```

#### Module Purposes

| Module    | Purpose                                    |
| --------- | ------------------------------------------ |
| cv2       | Computer vision and object detection       |
| numpy     | Mathematical operations and array handling |
| time      | Timing announcements                       |
| threading | Background speech processing               |
| queue     | Managing announcement tasks                |
| gtts      | Converting text to speech                  |
| pygame    | Playing generated audio files              |
| os        | File management and cleanup                |

---

## Required Model Files

The following files must be present in the project directory:

### MobileNet SSD Configuration

```text
deploy.prototxt
```

### MobileNet SSD Weights

```text
mobilenet_iter_73000.caffemodel
```

These files are loaded using:

```python
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'mobilenet_iter_73000.caffemodel'
)
```

---

## Project Structure

```text
AI-Object-Detection/
│
├── main.py
├── deploy.prototxt
├── mobilenet_iter_73000.caffemodel
├── README.md
```

---

## How It Works

1. The webcam captures live video.
2. Each frame is processed by the MobileNet SSD neural network.
3. Objects with confidence scores greater than 50% are detected.
4. Bounding boxes and labels are drawn around detected objects.
5. Every 5 seconds:

   * The program gathers all currently detected objects.
   * A speech announcement is generated.
   * The announcement is played through the computer speakers.
6. Press **Q** to quit the application.

---

## Running the Program

Run the script from the project directory:

```bash
python main.py
```

or

```bash
python3 main.py
```

---

## Example Output

Visual Output:

```text
Person: 92.45%
Chair: 87.12%
Bottle: 76.34%
```

Audio Output:

```text
Detected person
Detected chair
Detected bottle
```

---

## Requirements

* Python 3.9+
* Webcam
* Internet connection (required by gTTS to generate speech)
* Speakers or headphones

---

## Future Improvements

* Distance estimation for detected objects
* Directional guidance (left/right/center)
* Offline speech synthesis
* Improved object detection models such as YOLOv8
* Obstacle avoidance functionality
* Custom object training

---

## License

This project is intended for educational and research purposes.
