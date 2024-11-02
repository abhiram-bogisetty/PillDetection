import base64
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from django.shortcuts import render
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import io
import urllib.parse

# Create your views here.
def index(request):
    return render(request,'index.html')


def AdminAction(request):
    name = request.POST.get('username')
    apass = request.POST.get('password')
    if name == 'Admin' and apass == 'Admin':
        return render(request, "AdminApp/AdminHome.html")
    else:
        context = {'data': "Admin Login Failed..!!"}
        return render(request, "index.html", context)


def Adminhome(request):
    return render(request, "AdminApp/AdminHome.html")


def ViewAllUsers(request):
    con = sqlite3.connect("pilldetection.db")
    cur=con.cursor()
    cur.execute("select * from user")
    data=cur.fetchall()
    strdata="<table border=1><tr><th>Name</th><th>Email</th><th>Mobile</th><th>Address</th><th>Username</th><th>Action</th></tr>"
    for i in data:
        strdata+="<tr><td>"+str(i[1])+"</td><td>"+str(i[2])+"</td><td>"+str(i[3])+"</td><td>"+str(i[4])+"</td><td>"+str(i[5])+"</td><td><a href='/Delete?id="+str(i[0])+"'>Delete</a></td></tr>"
    context={'data':strdata}
    return render(request,'AdminApp/ViewAllUsers.html',context)

def Delete(request):
    uid=request.GET['id']
    con = sqlite3.connect("pilldetection.db")
    cur1=con.cursor()
    cur1.execute("delete from user where id='"+uid+"'")
    con.commit()
    cur=con.cursor()
    cur.execute("select * from user")
    data=cur.fetchall()
    strdata="<table border=1><tr><th>Name</th><th>Email</th><th>Mobile</th><th>Address</th><th>Username</th><th>Action</th></tr>"
    for i in data:
        strdata+="<tr><td>"+str(i[1])+"</td><td>"+str(i[2])+"</td><td>"+str(i[3])+"</td><td>"+str(i[4])+"</td><td>"+str(i[5])+"</td><td><a href='/Delete?id="+str(i[0])+"'>Delete</a></td></tr>"
    context={'data':strdata}

    context={'data':strdata}
    return render(request,'AdminApp/ViewAllUsers.html',context)




global traindataset,testdataset

def UploadDataset(request):

    global traindataset,testdataset

    traindataset = "Dataset\\train"
    testdataset = "Dataset\\valid"
    context = {'data':"Dataset Uploaded Successfully...!!"}
    return render(request, "AdminApp/UploadDataset.html", context)


global training_set, test_set
def DataGenerate(request):

    global training_set, test_set
    train_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory(traindataset,
                                                     target_size=(48, 48),
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     shuffle=True)
    test_set = test_datagen.flow_from_directory(testdataset,
                                                target_size=(48, 48),
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=False)

    context = {'data': 'Generated Training And Testing Images successfully','tr_size':len(training_set.filenames),'t_size':len(test_set.filenames)}
    return render(request, 'AdminApp/Generate.html', context)


def ClassIndices(traing):
    train_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(traing,
                                                     target_size=(48, 48),
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     shuffle=True)
    return training_set



global classifier

def GenerateCNN(request):
    global classifier
    if os.path.exists("Model\\model_weights.h5"):
        classifier = Sequential()
        classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="softmax", units=20))
        classifier.load_weights('Model/model_weights.h5')
        model=np.load('Model/my_history.npy',allow_pickle='TRUE').item()
        # summarize history for accuracy
        plt.plot(model['accuracy'])
        plt.plot(model['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(model['loss'])
        plt.plot(model['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        context = {"data": "CNN Model Loaded Successfully.."}
        return render(request, 'AdminApp/LoadModel.html', context)
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="softmax", units=20))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model = classifier.fit_generator(training_set,
                                 steps_per_epoch=100,
                                 epochs=50,
                                 validation_data=test_set,
                                 validation_steps=50)
        classifier.save_weights('Model/model_weights.h5')
        final_val_accuracy = model.history['accuracy'][-1]
        np.save('Model/my_history.npy', model.history)
        msg=f'Final Accuracy: {final_val_accuracy:.4f}'
        # summarize history for accuracy
        plt.plot(model.history['accuracy'])
        plt.plot(model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        context = {"data": "CNN Model Generated Successfully..","msg":msg}
        return render(request, 'AdminApp/LoadModel.html', context)


def logout(request):
    return render(request,'index.html')

