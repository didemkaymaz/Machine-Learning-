# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:07:40 2020

@author: DK
"""
import pandas as pd

names = ['index','duygu', 'haber'] 
data = pd.read_csv('aahaber.csv',engine='python',sep = ';', names=names)
#print(data)

x = data.iloc[:,2]
y = data.iloc[:,1]



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

# import numpy as np
# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

max_futures=1500
maxlen=622 
batch_size=40   
embedding_dims=53
epochs=3

from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

print('Model build..')
model=Sequential()
model.add(Embedding(max_futures, embedding_dims,input_length=maxlen))
    

model.add(Conv1D(embedding_dims, 8, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(8))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
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



