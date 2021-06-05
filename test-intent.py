import random
from keras.models import load_model
from string import punctuation
import nltk
import tensorflow as tf
from nltk.stem.snowball import SnowballStemmer as stemmer_fn
import numpy as np
import json

from tensorflow.keras.models import Sequential



with open('data/intents.json') as json_data:
    intents = json.load(json_data)

documents = []
words = []
context = {}
classes = sorted(list(set([intent['tag'] for intent in intents['intents']])))

stop_words = set(list(punctuation))
words = pre_process_words(words, stop_words)
words = sorted(list(set(words)))

LANG = 'english'
stemmer = stemmer_fn(LANG)
nltk.download('punkt',quiet=True)


def pre_process_words(wrds, stop):
    return [stemmer.stem(w.lower()) for w in wrds if w not in stop]

def pre_process_sentence(sentence, stop):
    wrds = nltk.word_tokenize(sentence)
    return [stemmer.stem(w.lower()) for w in wrds if w not in stop]

def bow_fn(sentence, words):
    wrds = pre_process_sentence(sentence, stop_words)
    bag = np.zeros((num_words))

    for i,w in enumerate(words):
        if w in wrds:
            bag[i] = 1
            
    return bag

            
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

model= tf.keras.models.load_model('backup/model2.h5')

num_words = len(words)


def inference(sentence, threshold, show_details=False):
    
    p = bow_fn(sentence, words)
    p = np.expand_dims(p,axis=0)

    results = model.predict(p)[0]
    y_pred = np.argmax(results)
    
    if show_details:
        print(results, y_pred)
    
    if results[y_pred] > threshold:
        return y_pred
    else:
        return None
    


text = 'Talk to you later jakub'

threshold = 0.3
#t = inference(text, threshold)
#print(classes[t])




def response(sentence, userID='user_ID', show_details=False):
    results = inference(sentence, threshold, show_details)

    if results is not None:
        intent_pred = classes[results]
        print(sentence)
        print(intent_pred)
        #for intent in intents['intents']:
        #    if intent['tag'] == intent_pred:
        #        if 'context_set' in intent:
        #            context[userID] = intent['context_set']

        #            if show_details: 
        #                print ('context:', intent['context_set'])

        #        # check if this intent is contextual and applies to this user's conversation
        #        if not 'context_filter' in intent or \
        #            (userID in context and 'context_filter' in intent and intent['context_filter'] == context[userID]):
        #            if show_details: 
        #                print('tag:', intent['tag'])

        #            return print(random.choice(intent['responses']))




context = {}
response("What's your education", userID='12345')
response("tell me more", userID='12345')
response("tell me more", userID='12345')
response('alright thanks', userID='12345')
response("cheers")
response("bye")

print('\n\n')
response("Where do you work", userID='15')
response("tell me more", userID='15')
response("tell me more", userID='15')
response('alright thanks', userID='15')
response("cheers")
response("bye")
response("hi")
