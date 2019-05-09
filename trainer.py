import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')

stemmer = LancasterStemmer()

# Load data from json file to array
with open('intents.json') as json_data:
    intents = json.load(json_data)

# ======================================================================
#               Json data to words, documents and classes
# ======================================================================

words = []
classes = []
documents = []
ignore_words = ['?']
# Loop through all intents from our intens array
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize every word
        w = nltk.word_tokenize(pattern)
        # Add the word with his token to our words list
        words.extend(w)
        # Add the word with his token to our documents with the tag
        documents.append((w, intent['tag']))
        # Add tag to our classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem all words in our words array and remove useless words
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words and w[0] is not "'"]

# Sort our words and classes
words = sorted(list(set(words)))
classes = sorted((list(set(classes))))

# ======================================================================
#                 Create more efficient training data
# ======================================================================

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    # Creates a new bag
    bag = []
    # Gets all words for a documentation (e.g. 'Hi', 'How are you', 'Hello', ... -> ['greeting'])
    pattern_words = doc[0]
    # Stemms all words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        # if pattern_words contains word (e.g. 'hi') append 1 else 0
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    # Creates an output where all classes are 1 if bag contains at least one 1
    output_row[classes.index(doc[1])] = 1
    # Training gets the bag and the ouput as one array
    training.append([bag, output_row])

# Shuffle data to use some of it as test data
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# ======================================================================
#                       Build the neural network
# ======================================================================

tf.reset_default_graph()

# Create the input layer with the length of the first element in train_x
net = tflearn.input_data(shape=[None, len(train_x[0])])
# Create two hidden layers with 8 nodes (back-propagations)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# Create the output layer which has the length of the first element in train_y and the action function softmax
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data', 'wb'))

# ======================================================================
#                  Load the build module (Could be extern)
# ======================================================================

data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

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
                    return print(random.choice(i['responses']))
            results.pop(0)


response('what are your hours today?')
