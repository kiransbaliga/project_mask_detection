import tensorflow
from keras.models import load_model
from keras_preprocessing import image
import numpy as np
import cv2,os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path='DataSets\Datasets'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))
print(label_dict)
print(categories)
print(labels)

img_size=100
data=[]
target=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:  
            resized=cv2.resize(img,(img_size,img_size))
           
            data.append(resized)
            target.append(label_dict[category])
       
        except Exception as e:
            print('Exception:',e)

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,3))
target=np.array(target)

new_target=np_utils.to_categorical(target)

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))

model.add(Dense(2,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_data,test_data,train_target,test_target=train_test_split(data,new_target,test_size=0.2)
history=model.fit(train_data,train_target,epochs=100,validation_split=0.2)

model.evaluate(test_data,test_target)
model.save('model.h5')
from tensorflow.keras.preprocessing import image
model=load_model('model.h5')
i=image.load_img('./Datasets/DataSets/Mask/1 (16).jpg',target_size=(100,100))
plt.imshow(i)
i=image.img_to_array(i)/255.0
i=i.reshape(1,100,100,3)
p=model.predict(i)
print(p)
print(p[0][0]>p[0][1])