import cv2
import os

def start_capture(name):
        path = "./data/" + name
        num_of_images = 0
        detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
        try:
            os.makedirs(path)
        except:
            print('Directory Already Created')
        vid = cv2.VideoCapture(0)
        while True:

            ret, img = vid.read()
            new_img = None
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                new_img = img[y:y+h, x:x+w]
            cv2.imshow("Face Detection", img)
            key = cv2.waitKey(1) & 0xFF


            try :
                cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
                num_of_images += 1
            except :

                pass
            if key == ord("q") or key == 27 or num_of_images > 300: #take 300 frames
                break
        cv2.destroyAllWindows()
        return num_of_images
#take frames by extract a video 
def take_video(name, video):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(video)
    if not vid.isOpened():
        print("Error: Could not open video file.")
        exit()
    num_of_images = 0
    while True:

        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        if not ret:
            break  # Break the loop if no more frames are available
        for x, y, w, h in face:
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            new_img = img[y:y+h, x:x+w]
        cv2.imshow("Face Detection", img)
        key = cv2.waitKey(1) & 0xFF
        try :
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except :

            pass
        if key == ord("q") or key == 27 or num_of_images > 300: #take 300 frames
            break
    vid.release()
    cv2.destroyAllWindows()
    return num_of_images


#take_video('tho1', 'data\WIN_20230920_07_56_11_Pro.mp4')
