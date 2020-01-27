#----------------------------------------------------------------#
#Modules
#----------------------------------------------------------------#
print("Initializing Modules...")
import os
import cv2
import numpy as np
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#Main Functions
#----------------------------------------------------------------#
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(CascadeClassifierFile)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        label = int(dir_name)
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        print(open(training_data+"/"+str(dir_name)+"/name", "r").read())
        for image_name in subject_images_names:
            try:
                if image_name == "name":
                    continue;
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                #cv2.imshow("Training Data", image)
                #cv2.waitKey(1)
                face, rect = detect_face(image)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
                print("   ",image_name)
            except:
                continue
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label, confidence = recognizer.predict(face)
    return img, face, rect, label, confidence

def new_recognizer(training_data,data_store):
    global faces
    global labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    files = sum([len(files) for r, d, files in os.walk(training_data)])
    if(os.path.isfile(data_store) and files == int(open(folder+"/files", "r").read())):
        print("Reading Model...")
        recognizer.read(data_store)
        labels = np.load(folder+"/labels.npy",allow_pickle=True).tolist()
        faces = np.load(folder+"/faces.npy",allow_pickle=True).tolist()
        print("Done Reading Model\n")
    else:
        print("Training Model:")
        faces, labels = prepare_training_data(training_data)
        np.save(folder+"/faces.npy", np.array(faces))
        np.save(folder+"/labels.npy", np.array(labels))
        try:
            recognizer.train(faces, np.array(labels))
        except:
            print("Error Training Model. Posibble because of no data.")
            raise SystemExit
        recognizer.write(data_store)
        open(folder+"/files", "w").write(str(files))
        print("Done Training Model\n")
    return recognizer
def add_data_to_recognizer(recognizer,face,label):
    global faces
    global labels
    faces.append(face)
    labels.append(label)
    np.save(folder+"/faces.npy", np.array(faces))
    np.save(folder+"/labels.npy", np.array(labels))
    recognizer.train(faces, np.array(labels))
    recognizer.write(data_store)
    open(folder+"/files", "w").write(str(sum([len(files) for r, d, files in os.walk(training_data)])))
    return recognizer

def fileName(before,after):
    number = 1
    while(os.path.isfile(before+str(number)+after)):
        number += 1
    return(before+str(number)+after)


#----------------------------------------------------------------#


#----------------------------------------------------------------#
#Variables
#----------------------------------------------------------------#
folder = "data" #Data Folder
training_data = folder+"/training-data" #Training Data Folder
data_store = folder+"/data.xml" #Model File
CascadeClassifierFile = folder+"/Cascade Classifiers/face.xml" #Cascade Classifier File
match = 40 #Classify if Less
skew = 5 # Range for Training
positive_skew = 5-skew #Extra Positive Range
learn = True #Further Train Model if Between Skew
save_unknown = False #Save Unknown Faces
exit = " " #Exit Charecter
#----------------------------------------------------------------#


#----------------------------------------------------------------#
#Main
#----------------------------------------------------------------#
if(not os.path.isdir(training_data)):
    os.mkdir(training_data)

recognizer = new_recognizer(training_data,data_store)

video_capture = cv2.VideoCapture(0)
name = ""
while True:
    try:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        img, face, rect, label, confidence = predict(frame)
        x,y,w,h = rect

        if(name != label and confidence < match):
            print("Found:",open(training_data+"/"+str(label)+"/name", "r").read())
            name = label

        if(confidence > match-skew and confidence < match+skew+positive_skew and learn):
            print("\nFuther Training Model...")
            cv2.imwrite(fileName(training_data+"/"+str(label)+"/",".jpg"),img[y-100:y+w+100, x-100:x+h+100])
            recognizer = add_data_to_recognizer(recognizer, face, label)
            print("Done Futher Training Model\n")

        label_text = open(training_data+"/"+str(label)+"/name", "r").read()+" - "+str(round(confidence))
        if(confidence > match):
            label_text = "Unknown("+label_text+")"
            if(name != "Unknown"):
                print("Unknown("+open(training_data+"/"+str(label)+"/name", "r").read()+")")
                name = "Unknown"
            if(save_unknown):
                if not os.path.exists("data/unknown"):
                    os.makedirs("data/unknown")
                cv2.imwrite(fileName(folder+"/unknown/",".jpg"),img[y-100:y+w+100, x-100:x+h+100])

        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord(exit):
            break
    except:
        if(name != "Empty"):
            print("Empty")
            name = "Empty"
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord(exit):
            break
video_capture.release()
cv2.destroyAllWindows()
#----------------------------------------------------------------#
