import mediapipe as mp
import cv2
import json
import datetime
import math
import threading

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # Calculate vectors between points
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # Calculate dot product and magnitude
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = (ba[0] ** 2 + ba[1] ** 2) ** 0.5
    magnitude_bc = (bc[0] ** 2 + bc[1] ** 2) ** 0.5

    # Calculate the angle in radians
    angle_rad = abs(math.acos(dot_product / (magnitude_ba * magnitude_bc)))

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Initialize MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Create dictionaries to store joint angle data for left and right sides
left_joint_angle_data = {
    "ankle": [],
    "knee": [],
    "hip": []
}

right_joint_angle_data = {
    "ankle": [],
    "knee": [],
    "hip": []
}

# Function to save joint angle data to separate JSON files
def save_data():
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save left side data
        with open("left_joint_angle_data.json", "w") as json_file:
            left_data = {"timestamp": current_time, "angles": left_joint_angle_data}
            json.dump(left_data, json_file, indent=4)

        # Save right side data
        with open("right_joint_angle_data.json", "w") as json_file:
            right_data = {"timestamp": current_time, "angles": right_joint_angle_data}
            json.dump(right_data, json_file, indent=4)

        threading.Event().wait(10)  # Save data every 10 seconds

# Start the auto-save thread
save_thread = threading.Thread(target=save_data)
save_thread.daemon = True
save_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            for side in ["left", "right"]:
                ankle_landmark = getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE")
                knee_landmark = getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE")
                hip_landmark = getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")

                ankle_angle = calculate_angle(
                    [landmarks[hip_landmark].x, landmarks[hip_landmark].y],
                    [landmarks[knee_landmark].x, landmarks[knee_landmark].y],
                    [landmarks[ankle_landmark].x, landmarks[ankle_landmark].y]
                )

                knee_angle = calculate_angle(
                    [landmarks[hip_landmark].x, landmarks[hip_landmark].y],
                    [landmarks[knee_landmark].x, landmarks[knee_landmark].y],
                    [landmarks[ankle_landmark].x, landmarks[ankle_landmark].y]
                )

                hip_angle = calculate_angle(
                    [landmarks[knee_landmark].x, landmarks[knee_landmark].y],
                    [landmarks[hip_landmark].x, landmarks[hip_landmark].y],
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER if side == "left" else mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER if side == "left" else mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                )

                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if side == "left":
                    left_joint_angle_data["ankle"].append({"timestamp": current_time, "angle": ankle_angle})
                    left_joint_angle_data["knee"].append({"timestamp": current_time, "angle": knee_angle})
                    left_joint_angle_data["hip"].append({"timestamp": current_time, "angle": hip_angle})
                else:
                    right_joint_angle_data["ankle"].append({"timestamp": current_time, "angle": ankle_angle})
                    right_joint_angle_data["knee"].append({"timestamp": current_time, "angle": knee_angle})
                    right_joint_angle_data["hip"].append({"timestamp": current_time, "angle": hip_angle})

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame with landmarks
    cv2.imshow("Joint Angle Monitoring", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
