import os
from dotenv import load_dotenv
import math
import numpy as np
import shutil
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras

load_dotenv(".env")

def dataFolder(p,split):
    if not os.path.exists("./"+p):
        os.mkdir("./"+p)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs("./"+p+"/"+dir)
            for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)),size=(math.floor(split*number_of_images[dir])-5),replace=False):
                O = os.path.join(ROOT_DIR,dir,img)
                D = os.path.join("./",p,dir)
                shutil.copy(O,D)
                os.remove(O)
    else:
        print(f"{p} folder exists")

ROOT_DIR = os.getenv("ROOT_DIR")
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR,dir)))


dataFolder("train",0.7)
dataFolder("val",0.15)
dataFolder("test",0.15)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu',input_shape = (224,224,3)))

model.add(Conv2D(filters=36, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss= keras.losses.binary_crossentropy, metrics=['accuracy'])

def preprocessImages1(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, rescale=1/255,horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224,224),batch_size=32, class_mode="binary")

    return image

path = "./train"
train_data = preprocessImages1(path)


def preprocessImages2(path):
    image_data = ImageDataGenerator(rescale=1/255)
    image = image_data.flow_from_directory(directory=path, target_size=(224,224),batch_size=32, class_mode="binary")

    return image

path = "./test"
test_data = preprocessImages1(path)

path = "./val"
val_data = preprocessImages1(path)

es = EarlyStopping(monitor="val_accuracy", min_delta=0.01,patience=6, verbose=1, mode='auto')

mc = ModelCheckpoint(monitor="val_accuracy", filepath="bestmodel.h5", verbose=1,save_best_only=True, mode='auto')

cd = [es,mc]

hs = model.fit_generator(generator=train_data, steps_per_epoch=8, epochs=30, verbose=1, validation_data=val_data, validation_steps=16, callbacks=cd)



model = load_model("bestmodel.h5")

acc = model.evaluate_generator(test_data)[1]

print(f"The accuracy of our model is {acc*100} %")

path = os.getenv("PATH")
img = load_img(path, target_size=(224,224))
input_array = img_to_array(img)/255

input_array = np.expand_dims(input_array,axis=0)
pred = model.predict(input_array)

if pred==0:
    print("The MRI is having a tumor")
else:
    print("The MRI is NOT having a tumor")
