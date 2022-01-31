# Creates a Depth map

import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading images
right_img = cv2.imread("data/gnome/right.jpg", cv2.IMREAD_GRAYSCALE)
left_img = cv2.imread("data/gnome/left.jpg", cv2.IMREAD_GRAYSCALE)
#right_img = cv.imread("data/gnome/right.jpg")
#left_img = cv.imread("data/gnome/right.jpg")

#getting height and width
height, width = right_img.shape[:2]
print(f"height: {height}")
print(f"width: {width}")

#resizing
scale_percent = 10 # percent of original size
width = int(right_img.shape[1] * scale_percent / 100)
height = int(right_img.shape[0] * scale_percent / 100)
dim = (width, height)

right_img_new = cv2.resize(right_img, dim)
left_img_new = cv2.resize(left_img, dim)

# creating disparity map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
depth = stereo.compute(left_img_new, right_img_new)

# cv.imshow("left", left_img)
# cv.imshow("right", right_img)

plt.imshow(depth)
plt.show()

# cv.waitKey(0)
# cv.destroyAllWindows()