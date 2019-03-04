import cv2
import os
import numpy as np

def facedetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=5)
    return faces,gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    for path,dirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipped system file")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img path",img_path)
            print("id",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                vontinue
            facees_Rect,gray_img=facedetection(test_img)
            if len(facees_Rect)!=1:
                continue# since we are assuming only single person images are being fed to class ClassName(object)
            (x,y,w,h)=facees_Rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID

def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
