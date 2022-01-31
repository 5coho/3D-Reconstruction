import cv2

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