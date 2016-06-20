from __future__ import print_function

'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

'''
Author: Gordon Erlebacher
Starting from lstm_text_generation.py, create a program to generate discrete signals. 
The signal can have up to "nval" values. 
The signal ranges between -1 and 1. 
Use same approach as char-rnn
Try this with RNN and LSTM. 
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy as sp
import random
import math
import sys
from collections import deque
import matplotlib.pyplot as plt

# is a vector of integers
def signal(x, nval):
    f = sp.sin(2.*x)
    mn, mx = min(f), max(f)
    dely = (mx - mn) / nval
    f1 = (f-mn) / dely
    return(f1.astype(int))


# signal to model (have neural network generate similar signal)
signal_length = 3000
xmin, xmax = 0., 100.
xx = np.linspace(xmin, xmax, signal_length)
x = signal(xx, nval=50)
#plt.plot(xx[0:400],x[0:400])
#plt.show()

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
#print('corpus length:', len(text))

# GE: C/R in litteracy text files are probably not important to generation of good text, unless
# \n is import for visual effect (such as writing linux programs). 
# Question: is weights or memory reset after "maxlen" characters? I would doubt it. 

chars = set(x)
#print(x)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# generate one-hot vectors for each character
encoded_x = np.zeros([len(chars), len(chars)])
for i in range(len(chars)):
    encoded_x[i,i] = 1
#print("----------------")
print(encoded_x)

# GE commentary
# cut the text in semi-redundant sequences of maxlen characters
# Does the stepping help when using LSTM? Not obvious
# Must plot decrease in loss function after each epoch
# Do stochastic gradients help?
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(x) - maxlen, step):
    sentences.append(x[i: i + maxlen])
    next_chars.append(x[i + maxlen])
#print('nb sequences:', len(sentences))
#for s in sentences:
    #print(s)

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    #print("sentence: ", sentence)
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=64, nb_epoch=10)
    #model.fit(X, y, validation_split=0.8, validation_data=None, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(x) - maxlen - 1)
    #sentence = deque(maxlen=maxlen)
    #print("sentence= ", sentence)

    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [1.0]:
        print()
        print('----- diversity:', diversity)

        generated = []
        # pick a random window to initiate random signal of length sig_len
        sentence = deque(x[start_index: start_index + maxlen], maxlen=maxlen)
        #generated.append(sentence)
        #print('----- Generating with seed: ', sentence)
        #sys.stdout.write(generated)
        #print(generated)


        sig_len = 400
        for i in range(sig_len):
            xx = np.zeros((1, maxlen, len(chars)))
            # transform sentence into one-hot vector
            for t, char in enumerate(sentence):
                xx[0, t, char_indices[char]] = 1.

            preds = model.predict(xx, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            #generated += next_char 
            # add new value to the sentence (integer)
            sentence.append(next_char) # = sentence[1:] + next_char

            #sys.stdout.write(next_char)
            #print(next_char)
            #sys.stdout.flush()
        plt.plot(sentence)
        plt.show()
