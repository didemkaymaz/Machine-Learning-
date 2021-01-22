# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:07:30 2020

@author: DK
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Activation, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import re
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from os import listdir
from os.path import isfile, join
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm


import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


#veri kümesi alındı. 
df  = pd.read_csv('aahaber.csv',engine='python', sep=';')#csv içine aktarmak için sep parametresi kullanıldı.
#Veri kümesine sütun adları verildi.
df_cols = ['index','Sentiment', 'SentimentText'] 
df.columns = df_cols

# Step - a : Varsa boş satırları kaldırır. 
df['SentimentText'].dropna(inplace=True)
# Step - b : Tüm metni küçük harf yapar.
df['SentimentText'] = [entry.lower() for entry in df['SentimentText']]
# Step - c : Tokenization: Bu şekilde, gruptaki her bir girdi girişindeki word_tokenize (giriş) sözcük kümesine bölünecektir.
df['SentimentText']= [word_tokenize(entry) for entry in df['SentimentText']]
# Step - d : Stop kelimeleri, Sayısal Olmayan ve perfom Word Stemming / Lemmenting'i kaldırır.
# WordNetLemmatizer, sözcüğün isim, fiil veya sıfat vb. Olup olmadığını anlamak için Pos etiketleri gerektirir. Varsayılan olarak Ad olarak ayarlanır 
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df['SentimentText']):
    # Bu adımın kurallarına uyan kelimeleri saklamak için Boş Liste Bildirme.
    Final_words = []
    # Aşağıdaki WordNetLemmatizer() başlatılıyor. 
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag fonksiyonu aşağıdaki 'etiketi' yani kelime İsim (N) veya Fiil (V) veya başka bir şey için, pos_tag (giriş) içindeki etiket: 
    for word, tag in pos_tag(entry):
         
        # Aşağıdaki koşul, kelimeleri stop sözcüklerini kontrol etmek ve içinde değilse sadece alfabeleri dikkate almaktır
        if word not in stopwords.words('turkish') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # Her yineleme için işlenen son kelime kümesi 'text_final'
    df.loc[index,'text_final'] = str(Final_words)
    
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['Sentiment'],test_size=0.70)

from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Tfidf_vect.vocabulary)
print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
SVM = svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
from sklearn.metrics import accuracy_score
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  
from sklearn import metrics

cm= confusion_matrix(Test_Y, predictions_SVM)  

cm=metrics.confusion_matrix(Test_Y,predictions_SVM,labels=[0,1,2]) 

df_cm=pd.DataFrame(cm,index=[i for i in [0,1,2]],columns=[i for i in [0,1,2]])

plt.figure(figsize=(7,5))
plt.title('SVM Confusion Matrix')
sns.heatmap(df_cm,annot=True,fmt='g')


