import cv2  # for the actual AI part
import numpy as np  # math functions for the AI part
import time  # for timing announcements
import threading  # for running announcements in a separate thread
import queue  # for queuing announcements
from gtts import gTTS  # google Text-to-Speech
import pygame  # for playing audio files
import os  # for file operations
# initialize pygame mixer for audio
pygame.mixer.init()
#queue for announcements
announcement_queue = queue.Queue()
# function to handle announcements in a separate thread
def tts_worker():
    while True:
        announcement = announcement_queue.get()
        if announcement is None:  # sentinel to stop the thread
            break
        try:
            # generate TTS audio file
            tts = gTTS(text=announcement, lang='en', slow=False)
            filename = f"temp_{int(time.time())}.mp3"
            tts.save(filename)
            # play audio file using pygame
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            #wait for playback to finish before cleaning up
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.remove(filename)
        except Exception as e:
            print(f"TTS error: {e}")
        announcement_queue.task_done()
# start the TTS worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()
# Load pre-trained MobileNet SSD model for object detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
# Class labels for COCO dataset
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "apple", "backpack", "banana",
           "baseball bat", "baseball glove", "bear", "bed", "bench", "book",
           "bowl", "broccoli", "cake", "carrot", "cell phone", "chair",
           "clock", "cup", "dining table", "donut", "elephant", "fork",
           "frisbee", "giraffe", "hair drier", "handbag", "horse", "hot dog",
           "keyboard", "kite", "knife", "laptop", "microwave", "motorcycle",
           "mouse", "orange", "oven", "parking meter", "pizza", "potted plant",
           "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink",
           "skateboard", "skis", "snowboard", "spoon", "sports ball", "stop sign",
           "suitcase", "surfboard", "teddy bear", "tennis racket", "tie", "toaster",
           "toilet", "toothbrush", "traffic light", "truck", "tv", "umbrella",
           "vase", "wine glass", "zebra"]
# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# Set up window for full screen display
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Initialize timer for announcements
last_announcement = time.time()
while True:
    # Capture frame by frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    # Prepare frame for neural network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    # Pass blob through the network
    net.setInput(blob)
    detections = net.forward()
    # Collect current frame's detected objects
    current_frame_objects = set()
    # Loop over detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.5:  
            idx = int(detections[0, 0, i, 1])
            current_frame_objects.add(idx) 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw green box and label
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Check if 5 seconds have passed since last announcement
    if time.time() - last_announcement >= 5:
        detected_classes = [CLASSES[idx] for idx in current_frame_objects]
        if detected_classes:
            # Queue each detected class for announcement
            for cls in detected_classes:
                announcement_queue.put(f"Detected {cls}")
        else:
            announcement_queue.put("No objects detected")
        last_announcement = time.time()
    # Display resulting frame
    cv2.imshow('Object Detection', frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Stop the TTS thread
announcement_queue.put(None)
tts_thread.join()
# Release capture and close windows
cap.release()
cv2.destroyAllWindows()