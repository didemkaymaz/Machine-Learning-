# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:04:49 2020

@author: DK

"""
import pandas as pd

names = ['Tweetler', 'Sınıf'] 
data = pd.read_csv('ortak.csv',engine='python',sep = ';', names=names)
#print(data)
data.Tweetler=data.Tweetler.astype(str)


x = [doc for doc in data.iloc[:,0]]
y = [doc for doc in data.iloc[:,1]]

from keras.preprocessing.text import Tokenizer    
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.utils import to_categorical

token = Tokenizer()
token.fit_on_texts(x)

x = token.texts_to_sequences(x)
x = pad_sequences(x)

x =StandardScaler().fit_transform(x)

y = preprocessing.LabelEncoder().fit_transform(y)
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =\
     train_test_split(x, y, test_size = 0.5)

max_futures=1500
maxlen=53 
batch_size=40   
embedding_dims=400
epochs=3

from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

print('Model build..')
model=Sequential()
#Pozitif tamsayıları (indeksler) sabit boyutlu dens vektörlerine dönüştür
#girdi özelliği alanını daha küçük bir alana sıkıştırmak için bir gömme katmanı kullanırız.
model.add(Embedding(max_futures, embedding_dims,input_length=maxlen))
    

model.add(Conv1D(embedding_dims, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D()) #maksimum havuzlama katmanı
# Girdi matrisi üzerinde bir pencere gezdirilirken her pencerenin en yüksek değeri o matrise karşılık gelen matrisi oluşturmaktadır.


model.add(Dense(3))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
t=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


la_ratio=model.evaluate(x_test,y_test)
print('Loss/Accuracy :', la_ratio)

from matplotlib import pyplot as plt


plt.plot(t.history['accuracy'],color='b', label="Training accuracy")
plt.plot(t.history['val_accuracy'],color='y', label="Test accuracy")
plt.plot(t.history['loss'],color='g', label="Training loss")
plt.plot(t.history['val_loss'],color='r', label="Test accuracy")
plt.title('Model')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy','Test Accuracy','Train Loss','Test Loss'],loc='bottom left')
print(plt.show())

from sklearn.metrics import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix
import numpy as np

rounded_pred = model.predict_classes(x_test, batch_size=128, verbose=0)
rounded_labels=np.argmax(y_test, axis=1)
print(confusion_matrix(rounded_labels,rounded_pred))
#print(plot_confusion_matrix(conf_mat=confusion_matrix(rounded_labels,rounded_pred)))



