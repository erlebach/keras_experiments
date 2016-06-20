# Input: (x,y) coordinates in the plane
# Output: Labels: 0 or 1

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.optimizers import SGD
import numpy as np
from pprint import pprint

# Create two concentric circles. Labels 0 are inside, label 1 is outside. 
# Uniform distribution of x,y
nb_points = 1000
radius = 1
frac = 0.8
pts = np.random.randn(nb_points, 2) # mean 0, std=1
labels = np.array([np.dot(p,p) for p in pts])
labels = np.where(labels < radius, 0, 1)
n1 = int(frac*nb_points)

from keras.utils.np_utils import to_categorical
nb_classes = 2
labels = to_categorical(labels, nb_classes)

X_train, X_test = pts[0:n1], pts[n1:]
Y_train, Y_test = labels[0:n1], labels[n1:]

model = Sequential()
model.add(Dense(8, input_dim=2,  activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

batch_size = 32


#model.compile(loss='binary_crossentropy',
model.compile(loss='categorical_crossentropy',
      optimizer='rmsprop',
      metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=100, batch_size=batch_size, shuffle=True)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print "score= ", score

y = model.predict_on_batch(X_test)
#y = y.reshape(np.prod(y.shape))

print y.shape, Y_test.shape
print "y[0], Y_test[0]= ", y[0], Y_test[0]

y1 = np.where(y < 0.5, 0, 1)
print "y1[0]= ", y1[0]

for i in xrange(len(y)):
	print y1[i], Y_test[i]

print "y-Y_test= ", (y1 - Y_test)
print "nb errors: ", (np.abs(y - Y_test)).sum()
