from PIL import Image
import matplotlib.pyplot as plt
# Load the image
file_path = './checkerboard1024-shaded.tif'
image = Image.open(file_path)

# Convert to grayscale
gray_image = image.convert('L')

# Convert to binary (black and white) using a global threshold
# Setting a threshold value; pixels above this value will be white, below will be black
threshold = 5
binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, '1')

# Save the binary image
binary_image_path = './checkerboard1024-binary.tif'
binary_image.save(binary_image_path)
plt.imshow(binary_image)