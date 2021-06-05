import random
import numpy as np
from string import punctuation

import nltk
from nltk.stem.snowball import SnowballStemmer as stemmer_fn

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import utils
from tensorflow.keras import layers

LANG = 'english'

stemmer = stemmer_fn(LANG)
nltk.download('punkt')

import json
with open('data/intents.json') as json_data:
    intents = json.load(json_data)

words = []
documents = []
classes = sorted(list(set([intent['tag'] for intent in intents['intents']])))
stop_words = set(list(punctuation))
            
# Go over the intents and their respective patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize patterns & skip stop words
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)

        # create pairs (tokenized sentence, intent)
        documents.append((wrds, intent['tag']))

'''
dictionary of words
- stemmed
- lowercase
- not in stop_words list
'''
words = pre_process_words(words, stop_words)
words = sorted(list(set(words)))


num_documents = len(documents)
num_classes = len(classes)
num_words = len(words)
num_classes = len(classes)

X = np.zeros((num_documents, num_words))
y = np.zeros((num_documents, num_classes))

# training set, bag of words for each sentence
for j,doc in enumerate(documents):
    wrds, intent = doc
    wrds = pre_process_words(wrds, stop_words)
    
    for i,w in enumerate(words):
        if w in wrds:
            X[j,i] = 1

    y[j,classes.index(intent)] = 1


idx = np.arange(num_documents)
random.shuffle(idx)

X = X[idx]
y = y[idx]

num_neurons = 10

model = Sequential()
model.add(Dense(num_neurons, input_shape=(X.shape[1],)))
model.add(Dense(num_neurons))
model.add(Dense(num_neurons))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(np.array(X), np.array(y), epochs=500, batch_size=8)


print("Model Training complete.")

#save the model
model.save("backup/model2.h5")

print("Model saved to backup folder.")

