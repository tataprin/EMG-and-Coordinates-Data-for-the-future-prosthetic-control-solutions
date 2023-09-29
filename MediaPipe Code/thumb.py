import mediapipe as mp
import cv2
import time
import json
import math
import numpy as np
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Set to 1 to detect only one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
json_file_path = 'thumb_data.json'
recording_interval = 0.2
data = []

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cos_theta)
    return math.degrees(angle)

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb = []

            for i in range(2, 6):  # Landmarks 2 to 5 correspond to the thumb
                x, y, z = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z
                thumb.append([x, y, z])

            thumb_angles = [
                calculate_angle(thumb[0], thumb[1]),
                calculate_angle(thumb[1], thumb[2]),
                calculate_angle(thumb[2], thumb[3])
            ]

            current_timestamp = datetime.now()

            data.append({
                "timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "thumb": {
                    "landmarks": thumb,
                    "angles": thumb_angles
                }
            })

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    current_time = time.time()

    if data and (current_time - datetime.strptime(data[-1]["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()) >= recording_interval:
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)

    cv2.imshow('Thumb Landmarks', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
