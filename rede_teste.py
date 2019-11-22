from keras.models import model_from_json
import numpy as np
from skimage.io import imread
import os
from skimage.transform import resize


def classificaValor(lista):
    for i in lista:
        vmax = max(i)
        indice = list(i).index(vmax)
        if indice == 0:
            print('cima')
        elif indice == 1:
            print('direita')
        elif indice ==2:
            print('esquerda')


#CAMINHO PARA ONDE O MODELO FOI SALVO
json_file = open('modelo1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("modelo.h5")
print("MODELO CARREGADO")


loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#CAMINHO ONDE AS IMAGENS DE TESTE ESTAO
path_img = "teste/"
files = os.listdir(path_img)
imagens = []
classes = []
for i in files:
    imagens = imagens + [imread(path_img + i)]

for i in range(len(imagens)):
    imagens[i] = resize(imagens[i], (imagens[i].shape[0] // 5, imagens[i].shape[1] // 5))


for i in range(len(imagens)):
    imagens[i] = imagens[i].flatten()

imagens = np.array(imagens)
imagens = imagens.reshape([-1,70, 128,1])

previsao = loaded_model.predict(imagens)
classificaValor(previsao)