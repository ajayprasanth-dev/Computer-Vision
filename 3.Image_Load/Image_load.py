import cv2
# Load the image
image = cv2.imread("test.jpg")
# Check if image is loaded successfully
if image is None:
 print("Error: Image not found or path is incorrect")
else:
 # Display the image
 cv2.imshow("Loaded Image", image)
 # Wait for a key press and close the window
 cv2.waitKey(0)
 cv2.destroyAllWindows()
