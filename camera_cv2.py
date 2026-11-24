import os  # for file operations
import queue  # for queuing announcements
import random  # for selecting varied spoken messages
import threading  # for running announcements in a separate thread
import time  # for timing announcements
from pathlib import Path  # for reliable model file paths

import cv2  # provided by the opencv-python package
import numpy as np  # math functions for the AI part
import pygame  # for playing audio files
from gtts import gTTS  # google Text-to-Speech
# Get this script's folder so model files load on Mac, Windows, or Linux
BASE_DIR = Path(__file__).resolve().parent

# initialize pygame mixer for audio
pygame.mixer.init()
# Use a Pygame fullscreen display for the camera feed. Pygame is already used
# for audio, and its fullscreen display avoids the OpenCV window title area.
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()
pygame.display.set_caption("Object Detection")
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
net = cv2.dnn.readNetFromCaffe(
    str(BASE_DIR / "deploy.prototxt"),
    str(BASE_DIR / "mobilenet_iter_73000.caffemodel")
)
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

# Alternative spoken messages for each class recognized by the MobileNet SSD model.
OBJECT_MESSAGES = {
    "aeroplane": [
        "Look up, there's an aeroplane!",
        "An aeroplane is flying by overhead!",
    ],
    "bicycle": [
        "A bicycle is here. Time for a ride?",
        "I see a bicycle. Let's get rolling!",
    ],
    "bird": [
        "A bird is nearby. Listen for its song!",
        "Look at that bird. I wonder where it's going!",
    ],
    "boat": [
        "There's a boat. Looks like a nice day on the water!",
        "I spotted a boat. All aboard!",
    ],
    "bottle": [
        "I spotted a bottle.",
        "There's a bottle nearby.",
    ],
    "bus": [
        "A bus is nearby. Maybe it's on its way to a stop!",
        "I see a bus. Hope everyone gets a seat!",
    ],
    "car": [
        "There's a car nearby. Watch out for traffic!",
        "I spotted a car. Safe travels!",
    ],
    "cat": [
        "A cat is here. What a cute little visitor!",
        "I see a cat. Someone's ready for a nap!",
    ],
    "chair": [
        "I found a chair. Take a load off!",
        "There's a chair nearby. A perfect place to sit!",
    ],
    "cow": [
        "A cow is nearby. Moo!",
        "I see a cow. Hope it's having a lovely day in the pasture!",
    ],
    "diningtable": [
        "There's a dining table. Is it time to eat?",
        "I spotted a dining table. Gather around!",
    ],
    "dog": [
        "A dog is here. Give your buddy a hello!",
        "I see a dog. What a good-looking pup!",
    ],
    "horse": [
        "There's a horse nearby. What a majestic sight!",
        "I spotted a horse. Ready for a gallop!",
    ],
    "motorbike": [
        "A motorbike is nearby. Hear that engine?",
        "I see a motorbike. Looking ready for the open road!",
    ],
    "person": [
        "Oh hey there, looking good today!",
        "Well hello! It's great to see you!",
    ],
    "pottedplant": [
        "I spotted a potted plant. It's looking leafy!",
        "There's a plant nearby. Someone has a green thumb!",
    ],
    "sheep": [
        "A sheep is nearby. Baa!",
        "I see a sheep. That's one fluffy neighbor!",
    ],
    "sofa": [
        "There's a sofa. Perfect for relaxing!",
        "I spotted a sofa. Save me a comfy seat!",
    ],
    "train": [
        "A train is nearby. All aboard!",
        "I see a train. Next stop, adventure!",
    ],
    "tvmonitor": [
        "I see a screen. Anything good on?",
        "There's a TV nearby. Time for a show?",
    ],
}

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
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
            # Queue a category-specific message, with a generic fallback for
            # any labels added to CLASSES later. A category can have one
            # message (string) or several alternatives (list/tuple).
            for cls in detected_classes:
                options = OBJECT_MESSAGES.get(cls)
                if options is None:
                    message = f"I see a {cls}."
                else:
                    message = random.choice(options)
                announcement_queue.put(message)
        else:
            announcement_queue.put("No objects detected")
        last_announcement = time.time()
    # Fit the camera frame inside the fullscreen Pygame window without cropping.
    frame_height, frame_width = frame.shape[:2]
    scale = min(screen_width / frame_width, screen_height / frame_height)
    fitted_width = int(frame_width * scale)
    fitted_height = int(frame_height * scale)
    resized_frame = cv2.resize(frame, (fitted_width, fitted_height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Pygame expects (width, height, channels); transpose axes without rotating
    # or mirroring the camera image.
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
    screen.fill((0, 0, 0))
    screen.blit(frame_surface, ((screen_width - fitted_width) // 2,
                                (screen_height - fitted_height) // 2))
    pygame.display.flip()

    # Break the loop on Q or Escape.
    should_quit = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
            should_quit = True
    if should_quit:
        break
# Stop the TTS thread
announcement_queue.put(None)
tts_thread.join()
# Release capture and close windows
cap.release()
pygame.quit()
