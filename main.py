import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from tensorflow import reshape
from sklearn.metrics import accuracy_score




def preprocessing_training_data(number_of_class, dataset_path):
    data = []
    labels =[]
    for i in range(number_of_class):
        path = os.path.join(dataset_path,'Train',str(i))
        images = os.listdir(path)#to wczyta wszystkie pliki z danego folderu
        for a in images:
            try:
                image = Image.open(path + '\\' + a)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(e)

    data = np.array(data)
    labels = np.array(labels)

    path_training = 'training'
    is_exist_data = os.path.exists(dataset_path + '/'+ path_training)
    if not is_exist_data:
        os.makedirs(path_training)
        print("dxd")

    np.save(dataset_path + '/'+ path_training + '/'+ 'data',data)
    np.save(dataset_path + '/'+ path_training + '/'+ 'target',labels)


def prepare_data_to_train(dataset_path, number_of_class):
    #Load data & Labels
    data = np.load(dataset_path +'/training/data.npy')
    labels = np.load(dataset_path + '/training/target.npy')

    print(data.shape, labels.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2,random_state=0)
    print(X_train.shape, X_test.shape,Y_train.shape,Y_test.shape)

    #One hot encoding - prościej sie nie da
    Y_train = to_categorical(Y_train,number_of_class)
    Y_test = to_categorical(Y_test,number_of_class)

    return X_train,X_test,Y_train, Y_test




def build_model(input_shape,number_of_class):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    # We have 43 classes that's why we have defined 43 in the dense
    model.add(Dense(number_of_class, activation='softmax'))



    return model


def plot_accuracy(history):
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()




def read_from_testcsv(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data = []
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test, label

def test_on_img(path_to_img,model):
    data = []
    image = Image.open(path_to_img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test = np.array(data)
    X_test = X_test[:,:,:,:3]
    print(X_test.shape)
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)



    return image,Y_pred


def learn_model():
    classes = 43
    os.chdir('C:\Datasety\German-classification')
    cur_path = os.getcwd()

    #Uczenie modelu
    preprocessing_training_data(classes, cur_path)
    X_train,X_test,Y_train, Y_test = prepare_data_to_train(cur_path,classes)
    model = build_model(X_train.shape[1:],classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 20
    history = model.fit(X_train,Y_train,batch_size=128,epochs=epochs, validation_data=(X_test,Y_test))
    plot_accuracy(history)
    model.save("./training/TSR.h5")

    return model

def test_model(model):
    # Na zbiorze testowym
    X_test, label = read_from_testcsv(cur_path+'/Test.csv')

    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred,axis=1)
    print(accuracy_score(label,Y_pred))

def auto_test_cropped_file_from_yolo(model):
    import classes
    path_to_cropped_img = 'C:\Traffic_Signs\cropped'
    path_to_log = 'C:\Traffic_Signs\log.txt'
    list_of_img = os.listdir(path_to_cropped_img)

    path = ''
    list = []
    for i in list_of_img:
        path = path_to_cropped_img + '\\' + i
        plot, prediction = test_on_img(path, model)
        s = [str(j) for j in prediction]  # konwersja z numparray koncowo do zwyklego inta
        a = int("".join(s))
        x = path, classes.classes[a]
        list.append(x)
    f = open(path_to_log, 'w')
    f.write(str(list))


if __name__ == "__main__":
    #Wczytac sciezke do zdjecia z kamery
    #Odpalic yolo z poziomu pythona
    #Wypluje to Jsona,
    #Przetworzyc Jsona, wyciac obraz
    #Obraz wrzucic na siec
    #Wynik klasyfikatora w konsoli

    #Na koniec jak bedzie dzialac, wszystkie sciezki ujednolicic

    os.chdir('C:\Datasety\German-classification')
    cur_path = os.getcwd()


    #Zaladowanie z istniejacych wag
    model = load_model(cur_path+'/training/TSR.h5')

    auto_test_cropped_file_from_yolo(model)



    # Z plików przycietych po YOLO:






























