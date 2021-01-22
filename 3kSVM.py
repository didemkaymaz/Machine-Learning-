# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:06:57 2020

@author: DK

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


sentences_training = []
classification_training = []

path='raw_texts/**/*.txt'

for sayi, tweetdosyasi in enumerate(glob(path, recursive=True)): 
    classification_training.append(tweetdosyasi.split("\\")[1]),
    sentences_training.append((open(tweetdosyasi, encoding="windows-1254").read().replace('\n', ' '))) 

print(sentences_training[0])


vectorizer = TfidfVectorizer(analyzer = "word", lowercase = True)
sent_train_vector = vectorizer.fit_transform(sentences_training)

print(sent_train_vector)

x_train, x_test, y_train, y_test = train_test_split(sent_train_vector.toarray(),classification_training, test_size=0.70)  



svm4 =  SVC(kernel='linear')
svm4.fit(x_train, y_train)
prediction4 = svm4.predict(x_test)
print("SVM (kernel=sigmoid) Accuracy :", accuracy_score(y_test, prediction4))  
print('SVM (kernel=sigmoid) Clasification: \n', classification_report(y_test, prediction4))  
print('SVM (kernel=sigmoid) Confussion matrix: \n', confusion_matrix(y_test, prediction4,labels=['1','2','3']))  
print("\n")
print("\n")
mat4 = confusion_matrix(y_test, prediction4)
names4 = np.unique(prediction4)
sns.heatmap(mat4, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=names4, yticklabels=names4)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Test size=0.7 iken kernel=linear')
plt.show()

