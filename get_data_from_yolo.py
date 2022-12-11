import cv2 as cv
import os
import json


def crop_image_by_coordinates(path_to_img,center_x, center_y, width, height,path_to_cropped_img):

    img = cv.imread(path_to_img)

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

    crop_img = img[y0:y1,x0:x1,:]

    cv.imwrite(path_to_cropped_img,crop_img)
    #cv.imshow("Display",img)
    #cv.waitKey(0)



def change_backslash_into_slash_in_file(path_to_file):
    with open(path_to_file,'r') as file:
        data = file.read()
        data = data.replace('\\','/')
    with open(path_to_file,'w')  as file:
        file.write(data)


def handle_json_from_yolo(path_to_json,path_to_save_cropped_img):
    change_backslash_into_slash_in_file(path_to_json)

    file = open(path_to_json)
    parsed_json = json.load(file)
    file_paths = []
    table_of_obj = []
    center_x = 0
    center_y = 0
    width = 0
    height = 0
    path_to_img = ""
    path_to_cropped_img_org = path_to_save_cropped_img

    for i in parsed_json:
        file_paths.append(i["filename"])
        table_of_obj.append(i["objects"])

    j = 0
    for i in table_of_obj:
        path_to_img = file_paths[j]
        l = len(i)
        for k in range(l):
            center_x = i[k]["relative_coordinates"]["center_x"]
            center_y = i[k]["relative_coordinates"]["center_y"]
            width = i[k]["relative_coordinates"]["width"]
            height = i[k]["relative_coordinates"]["height"]
            print(path_to_img, center_x, center_y, width, height)
            path_to_cropped_img = path_to_cropped_img_org + str(j) + '_' + str(k) + '.jpg'
            print(path_to_cropped_img)
            crop_image_by_coordinates(path_to_img, center_x, center_y, width, height, path_to_cropped_img)

        j += 1

if __name__ == "__main__":
    handle_json_from_yolo('result.json',"C:\Traffic_Signs\cropped/")







