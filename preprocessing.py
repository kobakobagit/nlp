# improt ===
!pip install stemming
!pip install nltk
import os
import nltk
import pandas as pd
import numpy as np
import re

# data settings ===
abs_path = '/content/drive/My Drive/signate/studentcup2020'
train_df = pd.read_csv(os.path.join(abs_path, 'data', 'train.csv'))
test_df  = pd.read_csv(os.path.join(abs_path, 'data', 'test.csv'))

train_index = train_df["id"] # id
train_ans = train_df["jobflag"] # target
train_pre = train_df["description"] # data

test_index = test_df["id"] # target
test_pre = test_df["description"] # data

# cleaning ===
def clearn(text):
    text = re.sub(r',', '', text)
    text = re.sub(r'\.', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'/', ' ', text)
    return text

for i in range(len(train_pre)):
  train_pre[i] = clearn(train_pre[i])

for i in range(len(test_pre)):
  test_pre[i] = clearn(test_pre[i])

print(train_pre[:2])
print(test_pre[:2])

# lemmatize_stemming ===
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.stem import SnowballStemmer

def lemmatize_stemming(text,output):
    for i in range(len(text)):
        sentence = text[i].split(' ')
        tmp = []
        for j in range(len(sentence)):
            word = sentence[j]
            word = lemma.lemmatize(word)
            tmp.append(stem.stem(word))
        output.append(tmp)

lemma = WordNetLemmatizer()
stem = SnowballStemmer(language='english')
train_pre_lemmatize_stemming = []
test_pre_lemmatize_stemming = []

lemmatize_stemming(train_pre, train_pre_lemmatize_stemming)
lemmatize_stemming(test_pre, test_pre_lemmatize_stemming)
print(train_pre_lemmatize_stemming[:2])
print(test_pre_lemmatize_stemming[:2])

# stop word ===
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
def stop_words(text):
    for i in range(len(text)):
        text[i] = [word for word in text[i] if word not in stop_words]

stop_words(train_pre_lemmatize_stemming)
stop_words(test_pre_lemmatize_stemming)

print(train_pre_lemmatize_stemming[:2])
print(test_pre_lemmatize_stemming[:2])

# pd ===
def to_df(text,output):
    for i in range(len(text)):
        make_sentence = str()
        for j in range(len(text[i])):
            make_sentence = make_sentence + text[i][j] + " "
    output.append(make_sentence)

train_after = []
test_after = []
to_df(train,train_after)
to_df(test,test_after)

train_after = pd.DataFrame(train_after)
train_after.columns = ["description"]
test_after = pd.DataFrame(test_after)
test_after.columns = ["description"]

print(train_after[:2])
print(test_after[:2])

