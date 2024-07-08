import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('./Module_2/Week_1/Data/dog.jpeg')

# Lightness method
gray_img_01 = (np.max(img, axis=2) + np.min(img, axis=2))/2
print(gray_img_01[0, 0])

# Average method
gray_img_02 = np.mean(img, axis=2)
print(gray_img_02[0, 0])

# Luminosity method
gray_img_03 = img[:, :, 0]*0.21 + img[:, :, 1]*0.72 + img[:, :, 2]*0.07
print(gray_img_03[0, 0])
