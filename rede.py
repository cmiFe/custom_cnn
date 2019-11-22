from skimage.io import imread
import pandas as pd
import os
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.models import model_from_json
from keras import regularizers
from keras.regularizers import l1

def classificaValor(lista):
    for i in lista:
        if i < 0.3:
            print()



# AQUI E ONDE VC COLOCA O PATH DA IMAGEM
path_img = "dataset3/"


files = os.listdir(path_img)

imagens = []
classes = []
for i in files:
    imagens = imagens + [imread(path_img + i)]
    classes = classes + [int(str(i.split("_")[0]))]


for i in range(len(imagens)):
    imagens[i] = resize(imagens[i], (imagens[i].shape[0] // 5, imagens[i].shape[1] // 5))





x_treino,x_teste,y_treino,y_teste = train_test_split(imagens,classes, test_size = 0.3, shuffle = True)
y_treino = to_categorical(y_treino,num_classes = 3)
y_teste = to_categorical(y_teste,num_classes = 3)
x_treino = np.array(x_treino)
y_treino = np.array(y_treino)
x_teste = np.array(x_teste)
y_teste = np.array(y_teste)



x_treino = x_treino.reshape([-1,70, 128,1])
x_teste = x_teste.reshape([-1,70, 128,1])


print(y_treino[0])
print(y_treino[1])
print(y_treino[2])
print(y_treino[3])
print(y_treino[4])



model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=(70,128,1),activity_regularizer=l1(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(16, kernel_size=3, activation='relu',activity_regularizer=l1(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_treino, y_treino, validation_data=(x_teste, y_teste), epochs=40,batch_size=10)
scores = model.evaluate(x_treino, y_treino, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("modelo.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelo.h5")
print("MODELO SALVO")


