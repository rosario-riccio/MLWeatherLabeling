import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import glob
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from dbMongo import *
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#path where there are csv files for training/test
src = "/home/rosario/Scrivania/MLcsvTraining"
#path where there are csv files for evaluate/prediction
src1 = "/home/rosario/Scrivania/MLcsvEvaluation"
#path where there are csv files for prediction
src2 = "/home/rosario/Scrivania/MLcsvPrediction"
#if flag is true the h5 file exists, otherwise it'll be created
flag1 = True
#if flag is true,the evaluation, otherwise prediction
flag2 = False

def main():
    global flag1
    global flag2
    class_weight = {}
    if not flag1:
        print("h5 file doesn't exist")
        try:
            countLabel = managedb.countLabelDB()
        except Exception as e:
            print("Error DB",str(e))
            sys.exit(1)
        class_weight.update({0:1.})
        for i in range(1,countLabel):
            class_weight.update({i:80.})
        print(class_weight)

        model = Sequential()

        model.add(Dense(100, input_dim=10, activation="relu",kernel_initializer = 'uniform'))
        model.add(Dropout(rate = 0.1))

        # Adding the second hidden layer
        model.add(Dense(activation='relu',units=100,kernel_initializer='uniform'))
        # Adding the output layer
        model.add(Dropout(rate = 0.1))
        # # Adding the third hidden layer
        model.add(Dense(activation='relu', units=100, kernel_initializer='uniform'))
        # # Adding the output layer
        model.add(Dropout(rate=0.1))

        model.add(Dense(countLabel, activation="softmax"))

        model.compile(optimizer=Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
        model.summary()
        try:
            all_files = glob.glob(src+"/*.csv")
            print(all_files)
            df_from_each_file = (pd.read_csv(f,usecols=["T2C","SLP", "WSPD10","WDIR10","RH2","UH","TC500","GPH500","CLDFRA_TOTAL","DELTA_RAIN","type"]) for f in all_files)
            concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
            concatenated_df = concatenated_df.sample(frac=0.5)
            dataset = concatenated_df.values
        except Exception as e:
            print("There are no csv file",str(e))
            sys.exit(1)

        y = dataset[:, 10]
        X = dataset[:, 0:10]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train_cat = to_categorical(y_train,num_classes=countLabel)
        y_test_cat  = to_categorical(y_test,num_classes=countLabel)
        print("-------------------------------------------------------------\n")
        #fit
        history = model.fit(X_train,y_train_cat,validation_data=(X_test,y_test_cat),epochs=30,batch_size=256,shuffle=True,verbose=1,class_weight=class_weight)
        print("-------------------------------------------------------------\n")
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        model.save('weatherML1.h5')
    else:
        print("h5 file exists")
        try:
            countLabel = managedb.countLabelDB()
        except Exception as e:
            print("Error DB", str(e))
            sys.exit(1)
        try:
            model = load_model('weatherML1.h5')
        except Exception as e:
            print("There are no h5 file", str(e))
            sys.exit(1)
        if flag2 == True:
            # evaluation
            try:
                all_files = glob.glob(src1 + "/*.csv")
                print(all_files)
                df_from_each_file = (pd.read_csv(f,usecols=["T2C","SLP", "WSPD10","WDIR10","RH2","UH","TC500","GPH500","CLDFRA_TOTAL","DELTA_RAIN","type"]) for f in all_files)
                concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
                dataset = concatenated_df.values
            except Exception as e:
                print("There are no csv file")
                sys.exit(1)
            y1 = dataset[:, 10]
            X1 = dataset[:, 0:10]
            y_cat1 = to_categorical(y1, num_classes=countLabel)
            results = model.evaluate(X1, y_cat1, verbose=1, batch_size=128)
            print(" ")
            print('test loss, test acc:', results)
        else:
            # prediction
            try:
                all_files = glob.glob(src2 + "/*.csv")
                if len(all_files)!=1:
                    print("There are too many files or zero file")
                    return
                print(all_files)
                df_from_each_file = (pd.read_csv(f,usecols=["LONGITUDE", "LATITUDE", "T2C", "SLP", "WSPD10", "WDIR10", "RH2","UH", "TC500", "GPH500", "CLDFRA_TOTAL", "DELTA_RAIN", "type"]) for f in all_files)
                concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
                dataset = concatenated_df.values
            except Exception as e:
                print("There are no csv file")
                sys.exit(1)
            lng = dataset[:, 0]
            lat = dataset[:, 1]
            y1 = dataset[:, 12]
            X1 = dataset[:, 2:12]
            y_cat1 = to_categorical(y1, num_classes=countLabel)
            prediction = model.predict_classes(X1, verbose=1)
            with open(src2 + "/out1.csv", "w") as f:
                fieldnames = ["lon", "lat", "class_id"]
                writer1 = csv.DictWriter(f, extrasaction='ignore', fieldnames=fieldnames)
                writer1.writeheader()
                for i in range(len(prediction)):
                    print(prediction[i], y_cat1[i])
                    if prediction[i] != 0:
                        writer1.writerow({"lon": lng[i], "lat": lat[i], "class_id": prediction[i]})

if __name__ == "__main__":
    main()
