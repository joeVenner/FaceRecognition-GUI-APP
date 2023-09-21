import cv2
from time import sleep
from PIL import Image
from tkinter import messagebox

def main_app(name):
        
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")
        cap = cv2.VideoCapture(0)
        pred = False
        while True:
            ret, frame = cap.read()
            #default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:


                roi_gray = gray[y:y+h,x:x+w]

                id,confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                if confidence > 50:
                    #if u want to print confidence level
                            #confidence = 100 - int(confidence)
                        pred = True
                        text = 'Recognized: '+ name.upper()
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                       
                       
                else:   
                        pred = False
                        text = "Unknown Face"
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)
                       
                        
            cv2.imshow("image", frame)


            if cv2.waitKey(20) & 0xFF == ord('q'):
                print(pred)
                if pred == True :
                    messagebox.showinfo('Congrat', 'You have already checked in')
                else:
                    messagebox.showerror('Alert', 'Please check in again')
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
