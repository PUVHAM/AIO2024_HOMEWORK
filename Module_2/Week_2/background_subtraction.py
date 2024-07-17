import numpy as np
import cv2

bg1_image = cv2.imread('./Module_2/Week_2/ImageData/GreenBackground.png', 1)
bg1_image = cv2.resize(bg1_image, (678, 381))

ob_image = cv2.imread('./Module_2/Week_2/ImageData/Object.png', 1)
ob_image = cv2.resize(ob_image, (678, 381))

bg2_image = cv2.imread('./Module_2/Week_2/ImageData/NewBackground.jpg', 1)
bg2_image = cv2.resize(bg2_image, (678, 381))

def compute_difference(bg_img, input_image):
    bg_img = bg_img.astype(float)
    input_image = input_image.astype(float)
    
    difference_single_channel = np.mean(np.abs(bg_img - input_image), axis=2)
    difference_single_channel = difference_single_channel.astype('uint8')
    
    return difference_single_channel

def compute_binary_mask(difference_single_channel):
    threshold = 25
    mask = difference_single_channel > threshold
    
    difference_binary = np.zeros_like(difference_single_channel, dtype=np.uint8)
    difference_binary[mask] = 255
    difference_binary = np.stack((difference_binary,)*3, axis=-1)
      
    return difference_binary

def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(bg1_image, ob_image)

    binary_mask = compute_binary_mask(difference_single_channel)
    
    output = np.where(binary_mask==255, ob_image, bg2_image)
    
    return output


output = replace_background(bg1_image, bg2_image, ob_image)
cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()