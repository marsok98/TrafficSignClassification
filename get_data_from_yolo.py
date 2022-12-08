import cv2 as cv
import os

if __name__ == "__main__":
    center_x = 0.583411
    center_y = 0.535315
    width = 0.031203
    height = 0.055614

    filename = "C:\Traffic_Signs\ts\00000.jpg"

    img = cv.imread(filename)
    cv.imshow("Display",img)
    cv.waitKey(0)