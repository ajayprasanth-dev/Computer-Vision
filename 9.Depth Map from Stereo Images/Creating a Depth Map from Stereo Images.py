import cv2
import numpy as np

# Load stereo images (left and right)
left_img = cv2.imread(r"Left.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(r"Right.png", cv2.IMREAD_GRAYSCALE)

# Ensure both images are loaded
if left_img is None or right_img is None:
    print("Error: Images not loaded. Check file paths.")
    exit()

# Create Stereo BM (Block Matching) matcher
stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=15)

# Compute disparity map
disparity = stereo.compute(left_img, right_img)

# Normalize for visualization
depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
depth_map = np.uint8(depth_map)

# Display the depth map
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
