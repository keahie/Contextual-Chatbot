import nltk
import numpy as np
import tflearn
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Create the input layer with the length of the first element in train_x
net = tflearn.input_data(shape=[None, len(train_x[0])])

# Create two hidden layers with 8 nodes (back-propagations)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

# Create the output layer which has the length of the first element in train_y and the action function softmax
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

with open('intents.json') as json_data:
    intents = json.load(json_data)

model.load('./model.tflearn')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, bow_words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bow_bag = [0] * len(bow_words)
    for s in sentence_words:
        for i, word in enumerate(bow_words):
            if word == s:
                bow_bag[i] = 1
                if show_details:
                    print('found in bag: %s' % word)
    return np.array(bow_bag)


context = {}
ERROR_THRESHOLD = .25


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence, user_id='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details:
                            print('conxtext: %s' % i['context_set'])
                        context[user_id] = i['context_set']
                    if 'context_filter' not in i or \
                            (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                        if show_details:
                            print('tag: %s' % i['tag'])
                    return print(random.choice(i['responses']))
            results.pop(0)


response('i want to rent a moped')
response('i want to rent it on another day')
response('thanks. good bye')
