# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:36:50 2020

@author: panay
"""
# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt


# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
import seaborn as sn


'''
#download stop words ---run once---
nltk.download('stopwords')
'''

# DATASET
DATASET_COLUMNS = ["target", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 40
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10
Test_Word= 'meet'

# KERAS
SEQUENCE_LENGTH = 40
EPOCHS = 1
BATCH_SIZE = 5

# SENTIMENT
POSITIVE = "Convict"
NEGATIVE = "Decoy"
#NEUTRAL = "Neutral"
SENTIMENT_THRESHOLDS = (0.5, 0.5)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"
#read file
df = pd.read_csv('./Decoy-Convict.csv', encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

#check the dataset
print("Dataset size:", len(df))
print(df.head(5))

#label the dataset
#decode_map = {0:"Fantasy" , 2:'Neutral', 4:"Non-Fantasy"}
decode_map = {0:"Decoy" , 1:"Convict"}
def decode_sentiment(label):
    return decode_map[label]

start = time.time()
df.target = df.target.apply(lambda x: decode_sentiment(x))
end = time.time()
print('Time used to label the dataset: ',"%.2f" % (end - start))

target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")

#Pre-process data
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

start = time.time()
df.text = df.text.apply(lambda x: preprocess(x))
end = time.time()
print('Time used to preprocess the dataset: ',"%.2f" % (end - start))

#split data
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))

start = time.time()
documents = [_text.split() for _text in df_train.text] 
end = time.time()
print('Time used to split the dataset: ',"%.2f" % (end - start))

#word2vector library
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)

w2v_model.build_vocab(documents)

#view the voc build
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)

start = time.time()
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
end = time.time()
print('Time used to train w2v the dataset: ',"%.2f" % (end - start))

#view similar words to given word
print('words that are directly corolated with: ',Test_Word)
print(w2v_model.most_similar(Test_Word))

#tokenize the text
start = time.time()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)
vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)
end = time.time()
print('Time used to tokenize the dataset: ',"%.2f" % (end - start))

#padding
start = time.time()
x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
end = time.time()
print('Time used to tokenize the dataset: ',"%.2f" % (end - start))

#Embedding
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)


#Build RNN
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# load the network weights
filename="./model.h5"
model.load_weights(filename)

#compile
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

#Predict
def decode_sentiment(score, include_neutral=False):
    if include_neutral:        
        #label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=False):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  

#Confusion matrix
y_pred_1d = []
y_test_1d = list(df_test.target)
scores = model.predict(x_test, verbose=1, batch_size=8000)
y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]

print(predict(" i will call then u will know im real"))
print(predict("have u been with a guy my age "))     


#Classification Report
print(classification_report(y_test_1d, y_pred_1d))

#Accuracy Score
print('Accuracy: ',accuracy_score(y_test_1d, y_pred_1d))

cnf_matrix = confusion_matrix(y_test_1d,y_pred_1d)
df_cm = pd.DataFrame(cnf_matrix, index = [i for i in "TF"],
                  columns = [i for i in "TF"])
sn.heatmap(df_cm, annot=True)





