import base64
import io
import urllib.parse

from django.shortcuts import render
import sqlite3
from django.core.files.storage import FileSystemStorage
import cv2
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from AdminApp.views import ClassIndices
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import io
import io as oi

# Create your views here.
def Register(request):
    return render(request,'UserApp/Register.html')

def index(request):
    return render(request,'index.html')

def logout(request):
    return render(request,'index.html')

def RegAction(request):
    name=request.POST['name']
    email=request.POST['email']
    mobile=request.POST['mobile']
    username=request.POST['uname']
    password=request.POST['pwd']

    con = sqlite3.connect("pilldetection.db")
    cur=con.cursor()
    #i=cur.execute("CREATE TABLE user (ID INTEGER PRIMARY KEY AUTOINCREMENT,name varchar(100),email varchar(100),mobile varchar(100),username varchar(100),password varchar(100))")
    i=cur.execute("insert into user values(null,'"+name+"','"+email+"','"+mobile+"','"+username+"','"+password+"')")
    con.commit()
    con.close()
    if i == 0:
        context = {'data':'Registration Failed...!!'}
        return render(request,'UserApp/Register.html',context)
    else:
        context = {'data':'Registration Successful...!!'}
        return render(request,'UserApp/Register.html',context)


def LogAction(request):
    username=request.POST.get('uname')
    password=request.POST.get('pwd')
    con = sqlite3.connect("pilldetection.db")
    cur=con.cursor()
    cur.execute("select *  from user where username='"+username+"'and password='"+password+"'")
    data=cur.fetchone()
    if data is not None:
        request.session['user']=username
        request.session['userid']=data[0]
        return render(request,'UserApp/UserHome.html')
    else:
        context={'udata':'Login Failed ....!!'}
        return render(request,'index.html',context)



def home(request):
    return render(request,'UserApp/UserHome.html')

def Profile(request):
    uid=str(request.session['userid'])
    con = sqlite3.connect("pilldetection.db")
    cur=con.cursor()
    cur.execute("select * from user where id='"+uid+"'")
    data=cur.fetchall()
    strdata="<table border=1><tr><th>Name</th><th>Email</th><th>Mobile</th><th>Address</th><th>Username</th></tr>"
    for i in data:
        strdata+="<tr><td>"+str(i[1])+"</td><td>"+str(i[2])+"</td><td>"+str(i[3])+"</td><td>"+str(i[4])+"</td><td>"+str(i[5])+"</td></tr>"
    context={'data':strdata}
    return render(request,'UserApp/ViewProfile.html',context)



def ModelGraphs(request):
    model=np.load('Model/my_history.npy',allow_pickle='TRUE').item()
    plt.plot(model['accuracy'])
    plt.plot(model['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig=plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri=urllib.parse.quote(string)

    model2=np.load('Model/my_history.npy',allow_pickle='TRUE').item()
    plt2.plot(model2['loss'])
    plt2.plot(model2['val_loss'])
    plt2.title('Model Loss')
    plt2.ylabel('loss')
    plt2.xlabel('epoch')
    plt2.legend(['Train', 'Validation'], loc='upper left')
    fig2=plt2.gcf()
    buf2 = oi.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    string2 = base64.b64encode(buf2.read())
    uri2=urllib.parse.quote(string2)
    context={'data':uri,'loss':uri2}
    return render(request, 'UserApp/ModelGraph.html' ,context)





def Upload(request):
    return render(request,'UserApp/UploadImage.html')

global filename, uploaded_file_url
def imageAction(request):
    global filename, uploaded_file_url
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        location = myfile.name
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
        cv2.imshow('uploaded Image', imagedisplay)
        cv2.waitKey(0)
    context = {'data': 'Test Image Uploaded Successfully'}
    return render(request, 'UserApp/UploadImage.html', context)



def Test(request):

    training_set=ClassIndices("Dataset/train")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imagetest = image.load_img(BASE_DIR + "/" + uploaded_file_url, target_size=(48, 48))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis=0)
    loaded_classifier = Sequential()
    loaded_classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
    loaded_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    loaded_classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
    loaded_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    loaded_classifier.add(Flatten())
    loaded_classifier.add(Dense(activation="relu", units=128))
    loaded_classifier.add(Dense(activation="softmax", units=20))
    loaded_classifier.load_weights('Model/model_weights.h5')
    pred = loaded_classifier.predict(imagetest)
    print(str(pred) + " " + str(np.argmax(pred)))
    predict = np.argmax(pred)
    print(training_set.class_indices)
    global msg;
    for x in training_set.class_indices.values():
        if predict == x:
            msg = list(training_set.class_indices.keys())[list(training_set.class_indices.values()).index(x)]
    print("predicted number: " + str(predict))
    imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
    oring = imagedisplay.copy()
    output = imutils.resize(oring, width=600)
    data = msg
    cv2.putText(output, data, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Pill Detected ", output)
    cv2.waitKey(0)
    global info;
    dataset=pd.read_excel("Dataset/dataset.xlsx")
    for index, row in dataset.iterrows():
        if str(row['Name']) == msg:
            info=row['Information']
    context = {'data': data,'info':info}
    return render(request,'UserApp/ImagePredction.html', context)
