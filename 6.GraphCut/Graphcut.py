!pip install opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from google.colab.patches import cv2_imshow

# --- Manual Image Input ---
uploaded = files.upload()
image_name = list(uploaded.keys())[0]
img = cv2.imread(image_name)

if img is None:
    print(f"Error: Could not load image '{image_name}'")
else:
    # --- 1. Define Bounding Box ---
    # (You might need to adjust these values based on your image)
    rect = (50, 50, 200, 150)  # (x, y, width, height)

    # --- 2. Create Mask ---
    mask = np.zeros(img.shape[:2], np.uint8)

    # --- 3. Initialize Background and Foreground Models ---
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # --- 4. Apply GrabCut ---
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # --- 5. Modify Mask to Get Foreground ---
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # --- 6. Apply Mask to Extract Foreground ---
    segmented_img = img * mask2[:, :, np.newaxis]

    # --- Display Results ---
    plt.figure(figsize=(10, 5))

    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()
