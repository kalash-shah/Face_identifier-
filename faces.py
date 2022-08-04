import numpy as np
import cv2
import pickle
#import pyttsx3

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
cap = cv2.VideoCapture(0)

"""
speaker = pyttsx3.init('sapi5')
voices = speaker.getProperty('voices')
voiceRate = 150
speaker.setProperty('rate', voiceRate)
"""
labels = {}
with open("labels.pickle", "rb") as f:
    _labels = pickle.load(f)
    labels = {v:k for k,v in _labels.items()}


while (True):
    #cap.red always gives 2 values  
    ret, frame = cap.read()
    # we need to convert the frame into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        # region of intrest or ROI meaning that it only needs our image and will cut all the other parts. kind of like cropping the image
         
        roi_gray = gray[y:y+h, x:x+w] #(ycord, ycord+height) and same for x
        #ROI for color image
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and  conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2

            cv2.putText(frame, name, (x,y), font,1,  color,stroke, cv2.LINE_AA )
            #speaker.say(labels[id_])
            #speaker.runAndWait()
        img_item = '7.png'

        cv2.imwrite(img_item, roi_color)

        color = [0, 255, 0] #BGR and not RGB
        stroke = 2 #thickness of stroke
    
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke) #just drawing a rectangle 
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)

    #displaying the frame
    cv2.imshow('frame', frame)
    #for exiting the window
    if cv2.waitKey(20) & 0xFF == ord('q'): 
        break

# releasing the capture once we dont need
cap.release()
cv2.destroyAllWindows()