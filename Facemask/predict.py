
from keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.h5')
model.make_predict_function()



# def predict(test_image):
#     detected_face, rect = detect_face(test_image)
#     resized_test_image = cv2.resize(detected_face, (121,121), interpolation = cv2.INTER_AREA)
#     label= eigenfaces_recognizer.predict(resized_test_image)
#     label_text = tags[label[0]]
#     if type(rect) != type(-1):
#         draw_rectangle(test_image, rect)
#         draw_text(test_image, label_text, rect[0], rect[1]-5)
#     else:
#         print("No face found")
#     return test_image, label_text

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = model.predict(i)
	if(p[0][0] > p[0][1]):
		return "Mask on"
	else:
		return "No Mask detected"
msk=0
nmsk=0
for i in range(1,30):
    img='DataSets/DataSets/Maskless/1 ('+str(i)+').jpg'
    p=predict_label(img)
    print(p)
    if p=="Mask on":
        msk+=1
    else:
        nmsk+=1
print("hai")
print(msk,nmsk)