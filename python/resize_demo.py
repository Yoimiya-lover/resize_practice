import cv2

img = cv2.imread("boqi.jpg")
print(img.shape[0],img.shape[1])

dim = (int(img.shape[1] * 2),int(img.shape[0] * 2))

resized = cv2.resize(img,dim,fx=2,fy=2,interpolation = cv2.INTER_NEAREST)

cv2.imshow("original image",img)
cv2.imshow("Resized image",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("boqi_resized.jpg",resized)