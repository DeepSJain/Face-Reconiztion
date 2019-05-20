import os
import cv2

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("data/Cascade Classifiers/face.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return img[y-100:y+w+100, x-100:x+h+100], faces[0]


TName = str(input("Name: ")).title()
name = ""
for number in os.listdir("data/training-data"):
    if(open("data/training-data/"+str(number)+"/name", "r").read() == TName):
        name = TName
        break
if(name == ""):
    number = 1
    while(os.path.isdir('data/training-data/'+str(number))):
        number += 1
    os.mkdir('data/training-data/'+str(number))
    open('data/training-data/'+str(number)+"/name", 'w').write(TName)
print("Number:",number)
video_capture = cv2.VideoCapture(0)
imageNumber = 1
while(os.path.isfile('data/training-data/'+str(number)+"/"+str(imageNumber)+".jpg")):
    imageNumber += 1
while True:
    try:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        face, rect = detect_face(frame)
        cv2.imshow('Face', face)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            cv2.imwrite("data/training-data/"+str(number)+"/"+str(imageNumber)+".jpg", face)
            imageNumber += 1
        draw_rectangle(frame, rect)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        try:
            cv2.destroyWindow("Face")
        except:
            continue

video_capture.release()
cv2.destroyAllWindows()
