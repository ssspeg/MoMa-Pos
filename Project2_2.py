import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = './scratch.png'
image = cv2.imread(image_path, 0)  # 0 to read image in grayscale mode

# Apply Gaussian bilateral filter for noise reduction
filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=10, sigmaSpace=10)

# Save the filtered image
output_path = './filtered_scratch.png'
cv2.imwrite(output_path, filtered_image)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Bilateral Filter Applied')
plt.axis('off')

plt.tight_layout()
plt.show()
adaptive_threshold_image = cv2.adaptiveThreshold(filtered_image, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
# Invert the image to get the text in black
adaptive_threshold_image = cv2.bitwise_not(adaptive_threshold_image)

# Save the thresholded image
threshold_output_path = './adaptive_threshold_scratch.png'
cv2.imwrite(threshold_output_path, adaptive_threshold_image)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(filtered_image, cmap='gray')
plt.title('Bilateral Filter Applied')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(adaptive_threshold_image, cmap='gray')
plt.title('Adaptive Thresholding Applied')
plt.axis('off')

plt.tight_layout()
plt.show()

# Define a kernel for the morphological operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

# Apply morphological opening operation
opened_image = cv2.morphologyEx(adaptive_threshold_image, cv2.MORPH_OPEN, kernel)

# Save the opened image
opening_output_path = './opening_scratch.png'
cv2.imwrite(opening_output_path, opened_image)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(adaptive_threshold_image, cmap='gray')
plt.title('Adaptive Thresholding Applied')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(opened_image, cmap='gray')
plt.title('Morphological Opening Applied')
plt.axis('off')

plt.tight_layout()
plt.show()

# Find contours in the opened image
contours, hierarchy = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on the area
min_contour_area = 25  # this threshold can be adjusted based on specific requirements
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Create an empty image for displaying the contours
contour_image = np.zeros_like(opened_image)

# Draw the large contours on the empty image
cv2.drawContours(contour_image, large_contours, -1, (255), thickness=cv2.FILLED)

# Save the contour image
contour_output_path = './contour_scratch.png'
cv2.imwrite(contour_output_path, contour_image)

# Display the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(opened_image, cmap='gray')
plt.title('Before Contour Filtering')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contour_image, cmap='gray')
plt.title('After Contour Filtering')
plt.axis('off')

plt.tight_layout()
plt.show()