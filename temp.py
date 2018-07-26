import cv2
image = cv2.imread('./temp.png', 0)
edges = cv2.Canny(image,200,300)
cv2.imwrite('./temp_out.png', edges)