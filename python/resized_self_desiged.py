import cv2
import numpy as np

img  = cv2.imread("boqi.jpg")

def resized_Nearest(src,dimy,dimx):
    fx = dimx/src.shape[1]
    fy = dimy/src.shape[0]
    print(fx,fy)
    print(src.shape[1],src.shape[0])
    print(dimx,dimy)
    dst = np.zeros((dimy,dimx,3),dtype = np.uint8)
    for i in range(dimy):
        for j in range (dimx):
            x = round(j / fx)
            y = round(i / fy)
            x = min(x,src.shape[1] - 1)
            y = min(y,src.shape[0] - 1)
            dst[i,j][0] = src[y,x][0]
            dst[i,j][1] = src[y,x][1] 
            dst[i,j][2] = src[y,x][2]
    
    return dst

def resized_inser_linear(src,dimy,dimx):
    fx = dimx / src.shape[1]
    fy = dimy / src.shape[0]
    dst = np.zeros((dimy,dimx,3),dtype = np.uint8)
    for i in range(dimy):
        for j in range(dimx):
            x = j / fx
            y = i / fy
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1,src.shape[1] - 1)
            y2 = min(y1 + 1,src.shape[0] -1 )
            Q11,Q21 = src[y1,x1],src[y1,x2]
            Q12,Q22  = src[y2,x1],src[y2,x2]
            R1 = (x - x1) * Q21 + (x2 - x) * Q11
            R2 = (x - x1) * Q22 + (x2 - x) * Q12

            dst[i,j] = (y - y1) * R2 + (y2 - y) * R1

    return dst 


if __name__ == "__main__":
    src = cv2.imread("boqi.jpg")
    fx = 2
    fy = 2
    dimx = int(src.shape[1] * fx)
    dimy = int (src.shape[0] * fy)
    dst1 = resized_Nearest(src,dimy,dimx)
    cv2.imshow("src",src)
    cv2.imshow("dst",dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("boqi_resized_self_nearest.jpg",dst1)

    dst2 = resized_inser_linear(src,dimy,dimx)
    cv2.imshow("src",src)
    cv2.imshow("dst",dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("boqi_resized_self_linear.jpg",dst2)

