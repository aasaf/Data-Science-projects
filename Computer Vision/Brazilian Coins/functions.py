#import libreries

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
import os
from keras.layers import Dropout
import numpy as np
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D, MaxPool2D, Flatten, GlobalAvgPool2D, GlobalMaxPool2D, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt



#Bounderies validation tests
def bounderies_validation(detected_circles):
    for (x,y,r) in detected_circles[0,:]:
            cv.circle(output,(x,y),2,3)
            #draw circle
            #cv.circle(output,(x,y),r,(0,255,0),3)
            #cv.circle(output,(x,y),2,(0,255,255),3)
            #cv.circle(output,(x,y),r,3)

        #To validate we are not cross the boundries
        #x and y is small
        if(x<r+10 and y<r+10):
            output = output[0:y+r+1, 0:x+r+1]
        #x and y is big
        elif(y+r+10>480 and x+r+10>640):
            output = output[y-r-1:480, x-r-1:640]
            #x is small and y is big

        elif(x<r+10 and y+r+10>480):
            output = output[y-r-1:480, 0:x+r+1]
            #x is big and y is small
        elif(x+r+10>640 and y<r+10 ):
            output = output[0:y+r+1, x-r-1:640]
        #y is big
        elif((y+r+10>480 and x-r-10>0 ) or (y+r+10>480 and x+r+10<640)):
            output = output[y-r-1:480, x-r-1:x+r+1]
            #y is small
        elif((y<r+10 and x-r-10>0) or (y<r+10 and x+r+10<640)):
            output = output[0:y+r+1, x-r-1:x+r+1]
        #x is big
        elif((x+r+10>640 and y-r-10>0) or( x+r+10>640 and y+r+10<480)):
            output = output[y-r-1:y+r+3, x-r-1:640]
        #x is small
        elif((x<r+10 and y-r-10>0 ) or (x<r+10 and y+r+10<480)):
            output = output[y-r-1:y+r+1, 0:x+r+1]
        else:

            output = output[y-r-1:y+r+1, x-r-1:x+r+1]
            
        

        #for visualize only
        #cv.imshow('output',output)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
            
    return output


#detect  circles in image files and crop a rectangle
def detect_classification_coins(image_files):

    for i in image_files:
        img = cv.imread(str(i))
        output = img.copy()
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray,5)
        circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,25,param1=10,param2=40,minRadius=30,maxRadius=45)
        print("detected circles:" + str(type(circles)))
        if (type(circles).__name__ == "NoneType"):
            continue
        detected_circles = np.uint16(np.around(circles))
        output = bounderies_validation(detected_circles)

        # Get target
        target = int(i.stem.split('_')[0])


        #choose your path for crop the coins in classification coins
        path = r'C:\Users\Asaf\Desktop\brazilian coins\valid_classification_dataset\training set\all_data_labels2'
        cv.imwrite(path +'\\' + i.stem + '.jpg', output)



#convert from image files to numpy array
def convert_image_files_to_array(images):
    arr =[]
    for i in images:
        test_image = image.load_img(i)
        test_image = image.img_to_array(test_image)
        test_image = cv.resize(test_image,(int(64),int(64)))
        arr.append(test_image)
    arr = np.array(arr)
    return arr



def build_model():

    # Initialising the CNN
    classifier = Sequential()

    classifier.add(Conv2D(16, 3, activation='relu', padding='same', input_shape=(64, 64, 3)) )
    classifier.add( MaxPooling2D(2) )
    classifier.add( Conv2D(32, 3, activation='relu', padding='same') )
    classifier.add(BatchNormalization())
    classifier.add( MaxPooling2D(2) )
    classifier.add( Conv2D(64, 3, activation='relu', padding='same') )
    classifier.add(BatchNormalization())
    classifier.add( MaxPooling2D(2) )
    classifier.add( Conv2D(128, 3, activation='relu', padding='same') )
    classifier.add(BatchNormalization())
    classifier.add( MaxPooling2D(2) )
    classifier.add( Conv2D(256, 3, activation='relu', padding='same') )
    classifier.add(BatchNormalization())
    # Transition between CNN and MLP
    classifier.add( GlobalAvgPool2D() )
    # MLP network
    classifier.add( Dense(256, activation='relu') )
    classifier.add(Dropout(0.5))
    classifier.add( Dense(units = 5, activation='softmax') )
    optim = Adam(lr=1e-3)
    classifier.compile(optim, 'categorical_crossentropy', metrics=['acc'])

    callbacks = [
        ReduceLROnPlateau(patience=5, factor=0.1, verbose=True),
        ModelCheckpoint('best.model', save_best_only=True),
        EarlyStopping(patience=12)
]
    return classifier, callbacks



# prediction coins by image path
def predict_coin(img_path, classifier):
    test_image = image.load_img(img_path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    predictors_list = []
    for i in result[0]:
        predictors_list.append(i)
    position = predictors_list.index(max(predictors_list))

    if position == 0:
        return 5
    if position == 1:
        return 10
    if position == 2:
        return 25
    if position == 3:
        return 50
    if position == 4:
        return 100



def extract_labels(input_path_all_labels):
    image_files_all_labels = list(input_path_all_labels.glob('*.jpg'))
    target_all_labels = []
    for i in image_files_all_labels:
        target_all_labels.append(int(i.stem.split('_')[0]))
    #for working with tensors
    target_all_labels = np.array(target_all_labels)
    y_labels = pd.DataFrame()
    y_labels['label'] = target_all_labels
    #dummies - categorical data
    y_labels =  pd.get_dummies(y_labels['label'])
    return y_labels, image_files_all_labels



def detect_many_coins_for_regression(image_files, classifier):
    sum_predicted =[]
    target_list =[]
    coin_number=0
    for i in image_files:
        sum_regression_per_image = 0
        img = cv.imread(str(i))
        output = img.copy()
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray,5)
        circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,25,param1=10,param2=40,minRadius=30,maxRadius=45)
        if (type(circles).__name__ == "NoneType"):
            continue
        detected_circles = np.uint16(np.around(circles))
        final_output  = bounderies_validation(detected_circles)

            # Get target
        target = int(i.stem.split('_')[0])
        path =  r'C:\Users\Asaf\Desktop\brazilian coins\valid_regression_dataset\compute_regression'


        cv.imwrite(path +'\\' + i.stem + str(coin_number) + '.jpg', final_output)
        coin_pred= predict_coin(path +'\\' + i.stem + str(coin_number)+ '.jpg', classifier)
        sum_regression_per_image  = coin_pred + sum_regression_per_image
        coin_number+=1


        sum_predicted.append(sum_regression_per_image)
        target_list.append(target)


    return sum_predicted,target_list


