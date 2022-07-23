import cv2
name=dict()
name['0']='Aishu'
name['1']='Amulya'
name['2']='Kiran'
name['3']='Sushi'
name['4']='VV'
name['Unknown Person']='Unknown Person'
tags = ['0', '1', '2', '3', '4']

def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]

def predict(test_image):
    detected_face, rect = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (121,121), interpolation = cv2.INTER_AREA)
    label= eigenfaces_recognizer.predict(resized_test_image)
    label_text = tags[label[0]]
    if type(rect) != type(-1):
        draw_rectangle(test_image, rect)
        draw_text(test_image, label_text, rect[0], rect[1]-5)
    else:
        print("No face found")
        label_text = "Unknown Person"
    return test_image, label_text

def draw_rectangle(test_image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(test_image, label_text, x, y):
    cv2.putText(test_image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def text_to_speech(label):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(name[label]+" is at the door")
    engine.runAndWait()
    return name[label]

eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()
eigenfaces_recognizer.read('eigenfaces_recognizer.xml')
# eigenfaces_recognizer.save('eigenfaces_recognizer.xml')
def mainn():
    test_image = cv2.imread("./Face/7.png")
    predicted_image, label = predict(test_image)
    text_to_speech(label)

# mainn()