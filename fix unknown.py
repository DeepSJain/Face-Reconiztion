import os
import cv2

if not os.path.exists("data/unknown"):
    os.makedirs("data/unknown")

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("face.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

for image in os.listdir("data/unknown"):
    try:
        img = cv2.imread("data/unknown/"+image)
        if(True):
            cv2.imshow('Unknown', img)
            cv2.waitKey(100)
            name = ""
            while(name == ""):
                TName = str(input("Name: ")).title()
                for number in os.listdir("data/training-data"):
                    if(open("data/training-data/"+str(number)+"/name", "r").read() == TName):
                        name = TName
                        break
                    if(name == ""):
                        print("Name not found")
            print("Number:",str(number),"\n")
            if(number != 0):
                imageNumber = 1
                while(os.path.isfile('data/training-data/'+str(number)+"/"+str(imageNumber)+".jpg")):
                    imageNumber += 1
                os.rename("data/unknown/"+image, 'data/training-data/'+str(number)+"/"+str(imageNumber)+".jpg")
            else:
                os.remove("data/unknown/"+image)
                print("a")
        else:
            os.remove("data/unknown/"+image)
            print("b")
    except Exception as e:
        os.remove("data/unknown/"+image)
        print("c")
