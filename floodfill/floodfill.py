import cv2
import numpy as np

'''
图像说明：
图像为二值化图像，255白色为目标物，0黑色为背景
要填充白色目标物中的黑色空洞
'''


def FillHole(imgPath, SavePath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE);
    _,im_in = cv2.threshold(im_in,127,255,cv2.THRESH_BINARY)
    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint必须是背景
    # isbreak = False
    # for i in range(im_floodfill.shape[0]):
    #     for j in range(im_floodfill.shape[1]):
    #         if (im_floodfill[i][j] == 0):
    #             seedPoint = (i, j)
    #             isbreak = True
    #             break
    #     if (isbreak):
    #         break
    # 得到im_floodfill
    cv2.imshow('origin',im_in)
    cv2.waitKey(0)
    seedPoint=(200,0)
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    cv2.imshow('floodfill',im_floodfill)
    cv2.waitKey(0)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    cv2.imshow('im_floodfill_inv',im_floodfill_inv)
    cv2.waitKey(0)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    # 保存结果
    cv2.imwrite(SavePath, im_out)
    cv2.imshow('im_out',im_out)
    cv2.waitKey(0)


if __name__ == '__main__':
    imgPath='floodfill.png'
    SavePath='out.png'
    FillHole(imgPath, SavePath)
