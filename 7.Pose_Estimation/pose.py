import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow
from google.colab import files
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Upload image
uploaded = files.upload()
image_name = list(uploaded.keys())[0]
image = cv2.imread(image_name)

# Process the image
results = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Check if pose landmarks are detected
if results.pose_landmarks:
    # Draw landmarks and connections on the image
    annotated_image = image.copy()
    # Access POSE_CONNECTIONS from mp.solutions.pose, not mp_pose
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # Display the annotated image using matplotlib for better visualization in Colab
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Pose Estimation")
    plt.axis('off')  # Hide axis ticks and labels
    plt.show()
else:
    print("No pose landmarks detected in the image.")
