# prompt: Basic Image Processing - Loading Images, Cropping, Resizing, Thresholding, Contour Analysis, Blob Detection AND I NEED A MANNUAL INPUT OF IMAGE 

import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

uploaded = files.upload()
image_name = list(uploaded.keys())[0]  # Get the uploaded image's filename

# Load the image
img = cv2.imread(image_name)

if img is None:
    print(f"Error: Could not load image '{image_name}'")
else:
    # Display the original image
    cv2_imshow(img)

    # Cropping
    x, y, w, h = 100, 50, 200, 150  # Define cropping region (x, y, width, height)
    cropped_img = img[y:y+h, x:x+w]
    cv2_imshow(cropped_img)


    # Resizing
    resized_img = cv2.resize(img, (300, 200))  # Resize to 300x200 pixels
    cv2_imshow(resized_img)

    # Thresholding (convert image to grayscale first)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    cv2_imshow(thresh_img)

    # Contour Analysis
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2_imshow(contour_img)

    # Blob Detection (using SimpleBlobDetector)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100 # Adjust as needed
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)
    blob_img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2_imshow(blob_img)
