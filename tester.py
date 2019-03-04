import cv2
import os
import numpy as np
import facerecog as fr


#faces,faceID=fr.labels_for_training_data('trainingimages') #geting the images from the folder
#face_recognizer=fr.train_classifier(faces,faceID)  #training the all images so diifent folders to diffenrt images
#face_recognizer.save('trainingdata.yml')   #the trained images data saving as yml
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingdata.yml') # testing with the loaded data no need of comparing images one more time
name={0:"vinod",1:"Nazia"}      # each folder as its name like 0 folder as vinod and 1 folder as nazia
cap=cv2.VideoCapture(1)
while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.facedetection(test_img)
    print("faces_detected",faces_detected)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
    resized_img=cv2.resize(test_img,(450,650))
    cv2.imshow("facedetection",resized_img)
    cv2.waitKey(10)
    for face in faces_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+h]
            label,confindence=face_recognizer.predict(roi_gray)
            print("confindence: ",confindence)
            print("label: ",label)
            fr.draw_rect(test_img,face)
            predicted_name=name[label]
            if confindence>70:
                continue
            fr.put_text(test_img,predicted_name,x,y)
            resized_img=cv2.resize(test_img,(450,650))
            cv2.imshow("facedetection",resized_img)
            if cv2.waitKey(10)==ord('q'):
                break
cap.release()
cv2.destroyAllWindows
