# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:05:44 2020

@author: DK
"""

#Future, Python'un dilde uyumsuz değişiklikler getiren yeni sürümlere geçişi kolaylaştırmayı amaçlamaktadır. 
#Glob,dosya aramada kullanılıyor.
from __future__ import print_function
from glob import glob

x = [] #textlerin listesi için
y = [] #dosya isimlerini tutuyor

#enumerate fonksiyonu ile girdileri numaralandırıp listeler.
for index, file in enumerate(glob('raw_texts/**/*.txt', recursive=True)): #dosya yollarını alt klasörlere kadar dönürüyor 
    y.append(file.split("\\")[1]), #split metodu cümleleri ayırmak için kullnılıyor burada ise dosya isimlerini birbirinden ayırmak için 1,2,3
    x.append((open(file, encoding="windows-1254").read().replace('\n', ' ').strip().lower())) 
    #encoding windows 1254 karakter seti kullanılmıştır
    #strip satır sonu ve tabları kaldırır
    
print("Toplam veri sayısı : ",len(y))
value1 = [i for i in y 
              if i in '1']
value2 = [i for i in y 
              if i in '2']
value3 = [i for i in 
              y if i in '3']
print("1.sınıf : ",len(value1))
print("2.sınıf : ",len(value2))
print("3.sınıf : ",len(value3))

#Kelimelerin morfolojik eklerini çıkarmak için bir işlem arayüzü. Bu süreç stemming olarak bilinir.
#re, aranılan bir içeriğin ilgili metin içerisinde olup olmadığını kontrol eder.

import nltk.stem as stemmer
import re

#kelimeleri ayrı ele alma fonksiyonu 
def splitIntoStem(message):
    return [removeNumeric(stripEmoji(singleCharacterRemove(removePunctuation
                                                           (removeHyperlinks
                                                            (removeHashtags
                                                             (removeUsernames
                                                              (stemWord(word)))))))) for word in message.split()]
#kelimeleri küçük harf yapar
def stemWord(tweet):
    return tweet.lower()

#kullanıcı adlarını kaldırır
def removeUsernames(tweet):
    return re.sub('@[^\s]+', '', tweet)

#hashtagleri kaldırır
def removeHashtags(tweet):
    return re.sub(r'#[^\s]+', '', tweet)

#linkleri kaldırır.
def removeHyperlinks(tweet):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)

#nümerik karakterleri kaldırır.
def removeNumeric(value):
    blist2 = [item for item in value if not item.isdigit()]
    blist3 = "".join(blist2)
    return blist3

#noktalama işaretlerini kaldırır
def removePunctuation(tweet):

    return re.sub(r'[^\w\s]','',tweet)

#karakterleri kaldırır
def singleCharacterRemove(tweet):
    return re.sub(r'(?:^| )\w(?:$| )', ' ', tweet)

#emojieri kaldırır
def stripEmoji(text):

    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return RE_EMOJI.sub(r'', text)

#kelimeleri tek tek ayırır
for i,k in enumerate(x):
    x[i] = " ".join(splitIntoStem(k)).split()
    
x[0]

from nltk.corpus import stopwords

#anlam içermeyen kelimeleri kaldırıyor 
def removeStopWords(x):
    
    filtered_stopwords = []
    filtered_stopwords_number = []
    
    stop_words = stopwords.words('turkish')

    stop_words.append("bir")
    stop_words.append("iki")
    stop_words.append("üç")
    stop_words.append("dört")
    stop_words.append("beş")
    stop_words.append("altı")
    stop_words.append("yedi")
    stop_words.append("sekiz")
    stop_words.append("dokuz")
    stop_words.append("on")
    
    print("stop_words : ",stop_words)
    
    for i in x:
        filtered_sentence = [w for w in i 
                                 if not w in stop_words]
        filtered_stopwords_number.append(filtered_sentence)                      # listeyi döndürüyor
        filtered_stopwords.append(" ".join(filtered_sentence))                   # stringleri döndürüyor
    
    return filtered_stopwords,filtered_stopwords_number                                       


x,filtered_stopwords_number = removeStopWords(x)


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
     train_test_split(x, y, test_size = 0.2)

max_futures=1500
maxlen=21
batch_size=40   
embedding_dims=400
epochs=3

from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

print('Model build..')
model=Sequential()
model.add(Embedding(max_futures, embedding_dims,input_length=maxlen))
    

model.add(Conv1D(embedding_dims, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(3))
model.add(Activation('softmax'))
    
model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
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


