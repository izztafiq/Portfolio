#Import-------------------------------------------------------------------

import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from imutils.video import VideoStream

#Window-------------------------------------------------------------------

window = tk.Tk()
window.title("Face_Recogniser")
window.geometry("520x440")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'

window.configure(background='#383255')


#Label-------------------------------------------------------------------


lbl = tk.Label(window, text="Flight Number",width=10  ,height=1  ,fg="red"  ,bg="#b3eef5" ,font=('times', 15, ' bold ') ) 
lbl.place(x=250, y=10)

txt = tk.Entry(window,width=15  ,bg="#b3eef5" ,fg="red",font=('times', 15, ' bold '))
txt.place(x=400, y=10)
lbl = tk.Label(window, text="Passport Number",width=10  ,height=1  ,fg="red"  ,bg="#b3eef5" ,font=('times', 15, ' bold ') ) 
lbl.place(x=250, y=60)

txt = tk.Entry(window,width=15  ,bg="#b3eef5" ,fg="red",font=('times', 15, ' bold '))
txt.place(x=400, y=60)

lbl2 = tk.Label(window, text="Enter Name",width=10  ,fg="red"  ,bg="#b3eef5"    ,height=1 ,font=('times', 15, ' bold ')) 
lbl2.place(x=250, y=110)

txt2 = tk.Entry(window,width=15  ,bg="#b3eef5"  ,fg="red",font=('times', 15, ' bold ')  )
txt2.place(x=400, y=110)

lbl3 = tk.Label(window, text="Notification : ",width=15  ,fg="red"  ,bg="#b3eef5"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=250, y=200)

message = tk.Label(window, text="" ,bg="#b3eef5"  ,fg="red"  ,width=30  ,height=2, activebackground = "#c9f7c1" ,font=('times', 15, ' bold ')) 
message.place(x=420, y=200)

lbl3 = tk.Label(window, text="Attendance : ",width=15  ,fg="red"  ,bg="#b3eef5"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=250, y=300)


message2 = tk.Label(window, text="" ,fg="red"   ,bg="#b3eef5",activeforeground = "#c9f7c1",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=420, y=300)


#Function -------------------------------------------------------------
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

#Image---------------------------------------------------------------

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id=''
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),6)
                cv2.putText(im,"UKNOWN",(x,y+h), font, 1,(0,0,255),2)
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(0,225,0),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)


clearButton = tk.Button(window, text="Clear", command=clear  ,fg="red"  ,bg="#b3eef5"  ,width=10  ,height=1 ,activebackground = "#c9f7c1" ,font=('times', 15, ' bold '))
clearButton.place(x=600, y=10)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="red"  ,bg="#b3eef5"  ,width=10  ,height=1, activebackground = "#c9f7c1" ,font=('times', 15, ' bold '))
clearButton2.place(x=600, y=60)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="red"  ,bg="#b3eef5"  ,width=10  ,height=1, activebackground = "#c9f7c1" ,font=('times', 15, ' bold '))
takeImg.place(x=10, y=200)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"  ,bg="#b3eef5"  ,width=10  ,height=1, activebackground = "#c9f7c1" ,font=('times', 15, ' bold '))
trainImg.place(x=10, y=250)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="red"  ,bg="#b3eef5"  ,width=10  ,height=1, activebackground = "#c9f7c1" ,font=('times', 15, ' bold '))
trackImg.place(x=10, y=300)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red"  ,bg="#b3eef5"  ,width=5  ,height=1, activebackground = "#c9f7c1" ,font=('times', 15, ' bold '))
quitWindow.place(x=30, y=350)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Izzat")
copyWrite.configure(state="disabled",fg="red"  )
copyWrite.pack(side="left")
copyWrite.place(x=500, y=350)


window.mainloop()




