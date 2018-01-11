#encoding:utf-8

#
#图像二值化及反转
#
#(0, 0) --> (35, 161) set to 0
import numpy as np
import cv2
import MaUtilities as mu
from PIL import Image
mu.display("SmallImage/1.png")
for i in range(1, 221):
    image = cv2.imread("SmallImage/%d.png" % i)
    for x in range(161):
        for y in range(30):
            image[x][y] = 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#将图像转为灰色
    mu.show_detail(image)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)#高斯滤波
    #mu.display(blurred)
    (T, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)#阈值化处理，阈值为：155
    Image.fromarray(thresh).save("P1_SmallImage/%d.png" % i)
    print(i)
    #mu.display(thresh)

