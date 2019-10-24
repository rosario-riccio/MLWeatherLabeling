import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import glob
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from dbMongo import *

src = "/Users/rosarioriccio/Desktop/dataTemp/"

def main():
    countLabel = managedb.countLabelDB()
    model = Sequential()
    model.add(Dense(50, input_dim=10, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(countLabel, activation="softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()
    for filenamePath in Path(src).glob('**/*.csv'):
        filename = str(filenamePath)
        print("\n\n")
        print(filename)
        df = pd.read_csv(filename, usecols=["T2C","SLP", "WSPD10","WDIR10","RH2","UH","TC500","GPH500","CLDFRA_TOTAL","DELTA_RAIN","type"])
        y = df.type
        X = df.drop("type",axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train_cat = to_categorical(y_train,num_classes=countLabel)
        y_test_cat  = to_categorical(y_test,num_classes=countLabel)
        model.fit(X_train,y_train_cat,validation_data=(X_test,y_test_cat),epochs=50,batch_size=256,shuffle=True,verbose=1)
    model.save('path_to_my_model.h5')

if __name__ == "__main__":
    main()