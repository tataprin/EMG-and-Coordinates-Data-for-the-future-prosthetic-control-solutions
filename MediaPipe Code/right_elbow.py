import mediapipe as mp
import json
import cv2
import math
import time
from datetime import datetime

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the MediaPipe Pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(radians)
    return angle

# Function to save data to a JSON file and auto-update the JSON file
def save_data_to_json(data):
    with open('right_elbow_angle_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Initialize the data array
data = []

# Capture webcam input (you can replace this with your input source)
cap = cv2.VideoCapture(0)

# Initialize a timer
start_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Assuming that landmark 12 is the right shoulder
        # Assuming that landmark 14 is the right wrist
        # Assuming that landmark 13 is the right elbow
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        right_wrist = [landmarks[14].x, landmarks[14].y]
        right_elbow = [landmarks[13].x, landmarks[13].y]

        # Calculate the angle
        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Get the current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append the data to the list with current time
        data.append({
            'timestamp': current_time,
            'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
            'angle': angle,
            'right_shoulder': right_shoulder,
            'right_wrist': right_wrist,
            'right_elbow': right_elbow
        })

        # Draw the landmarks and angle on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image, f'Angle: {angle:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check the elapsed time and print the data every 0.2 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.2:
            print(data[-1])  # Print the latest data point
            start_time = time.time()  # Reset the timer

        # Save the data to the JSON file after each update
        save_data_to_json(data)

    cv2.imshow('Elbow Angle Estimation', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Save the final data to the JSON file
save_data_to_json(data)

# Close all OpenCV windows
cv2.destroyAllWindows()
