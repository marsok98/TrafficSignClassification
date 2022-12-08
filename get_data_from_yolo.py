import cv2 as cv
import os

if __name__ == "__main__":
    center_x = 0.306159
    center_y = 0.653138
    width = 0.034284
    height = 0.080505

    filename = "C:\Traffic_Signs/ts/00001.jpg"

    img = cv.imread(filename)

    y_size,x_size,_ = img.shape

    center_x_scaled = x_size * center_x
    center_y_scaled = y_size * center_y
    width_scaled = x_size * width
    height_scaled = y_size * height

    x0 = center_x_scaled-width_scaled/2
    y0 = center_y_scaled-height_scaled/2
    x0 = int(x0)
    y0 = int(y0)

    x1 = x0+width_scaled
    y1 = y0+height_scaled
    x1 = int(x1)
    y1 = int(y1)

    print(y0,y1)


    crop_img = img[y0:y1,x0:x1,:]

    cv.imwrite('C:\Traffic_Signs/cropped/00001.jpg',crop_img)



    #cv.imshow('',crop_img)
    #cv.waitKey(0)

    #cv.rectangle(img, xy,wh,(255,0,0),-1)

    #cv.imshow('Bounding Box',img)
    #cv.waitKey(0)







    cv.imshow("Display",img)
    cv.waitKey(0)
