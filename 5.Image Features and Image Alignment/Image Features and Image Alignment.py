import cv2
import numpy as np

# Load image
img = cv2.imread("images.jpg", cv2.IMREAD_COLOR)
if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Fourier Transform
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
normalized_spectrum = np.uint8(magnitude_spectrum / np.max(magnitude_spectrum) * 255)
cv2.imshow("Fourier Magnitude Spectrum", normalized_spectrum)

# 2. Hough Transform for line detection
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
hough_img = img.copy()
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("Hough Line Detection", hough_img)

# 3. ORB Feature Detection
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray, None)
img_orb = cv2.drawKeypoints(img, kp1, None, color=(0, 255, 0), flags=0)
cv2.imshow("ORB Features", img_orb)

# 4. Feature Matching and Alignment (self-matching for demo)
kp2, des2 = orb.detectAndCompute(gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
match_img = cv2.drawMatches(img, kp1, img, kp2, matches[:10], None, flags=2)
cv2.imshow("Feature Matching", match_img)

# 5. Simple Cloning based on ROI
clone = img.copy()
src_region = img[50:150, 50:150]  # Source region
clone[200:300, 200:300] = src_region  # Paste into target location
cv2.imshow("Image Cloning", clone)

cv2.waitKey(0)
cv2.destroyAllWindows()
