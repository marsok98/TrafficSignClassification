import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import file_path
import get_data_from_yolo
import yolo3image
import model_functions
import os
from tensorflow.keras.models import load_model
import classes
def delete_all_files_from_directory(folder_path):
    for file_name in os.listdir(folder_path):
        file = folder_path + file_name
        if os.path.isfile(file):
            os.remove(file)

def crop_img_from_obj_list(list_of_obj ,path_org_img, path_cropped_folder):
    img = cv2.imread(path_org_img)
    x_min = 0
    y_min = 0
    box_width = 0
    box_height = 0
    list_of_filenames = []
    l = len(list_of_obj)
    delete_all_files_from_directory(path_cropped_folder)
    for k in range(l):
        x_min = list_of_obj[k][1]
        y_min = list_of_obj[k][2]
        box_width = list_of_obj[k][3]
        box_height = list_of_obj[k][4]
        crop_img = img[y_min:y_min + box_height, x_min: x_min + box_width]
        path_to_save_cropped = path_cropped_folder + str(k) + '.jpg'
        cv2.imwrite(path_to_save_cropped, crop_img)
        list_of_filenames.append(path_to_save_cropped)
    return list_of_filenames

def make_predictions_and_draw(path_to_test):
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (128, 0, 255)]
    img_org = cv2.imread(path_to_test)
    img = img_org
    list = yolo3image.yolo3(path_to_test)
    paths = crop_img_from_obj_list(list, path_to_test, file_path.to_save_cropped_img_folder)
    model = load_model(file_path.network_weights)

    for k in range(len(paths)):
        x_min, y_min, box_width, box_height = list[k][1], list[k][2], list[k][3], list[k][4]

        _, Y_pred, confidence = model_functions.test_on_img(paths[k], model)

        cv2.rectangle(img, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colours[k], 2)
        text_box_current = '{}: {:.2f}'.format(classes.classes[Y_pred],
                                               confidence)
        cv2.putText(img, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colours[k], 2)

    path_to_save = file_path.results + 'result.jpg'
    #cv2.imshow("Predykcja", img)
    cv2.imwrite(path_to_save,img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
































