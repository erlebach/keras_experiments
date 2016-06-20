# Input: (x,y) coordinates in the plane
# Output: Labels: 0 or 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.patches as mpatches  # draw circles
from matplotlib.collections import PatchCollection

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import *
import keras
from pprint import pprint

# Create two concentric circles. Labels 0 are inside, label 1 is outside. 
# Uniform distribution of x,y
nb_points = 5000
radius = 1
frac = 0.8
np.random.seed(1000)
pts = np.random.randn(nb_points, 2) # mean 0, std=1
labels = np.array([np.dot(p,p) for p in pts])
labels = np.where(labels < radius, 0, 1)
#print zip(labels, rlabels)
n1 = int(frac*nb_points)

X_train, X_test = pts[0:n1], pts[n1:]
Y_train, Y_test = labels[0:n1], labels[n1:]

model = Sequential()
model.add(Dense(64, input_dim=2,  activation='relu', name='ge_dense1'))
model.add(Dropout(0.5, name='ge_dropout1'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
      optimizer='rmsprop',
      metrics=['accuracy'])

print model.summary()

# Shuffle=True might only be shuffling X and not y. Must test
progbar_logger = ProgbarLogger()
filepath = "./checkpoint.{epoch:02d}--{loss:.2f}--{acc:.2f}.hdf5"
filepath = "./checkpoint.{epoch:02d}"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=True, save_best_only=False, mode='auto')

print "-----------------------------"
# no output if verbose=False
#history = model.fit(X_train, Y_train, nb_epoch=10, batch_size=32, shuffle=True, verbose=False)
#print "-----------------------------"
# no output if verbose=False
history = model.fit(X_train, Y_train, nb_epoch=10, batch_size=32, shuffle=True, verbose=False, callbacks=[checkpoint])
print history.__dict__
print progbar_logger.__dict__
print "-----------------------------"
quit()
print dir(history)
print history.__dict__
print "progbar.__dict__: ", progbar_logger.__dict__
print history.epoch
 #'epoch', 'history', 'model', 'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end', 'on_train_begin', 'on_train_end', 'params']
quit()
#---------------------------------------------------
score = model.evaluate(X_test, Y_test, batch_size=32)
print "score= ", score
#---------------------------------------------------

y = model.predict_on_batch(X_test)
y = y.reshape(np.prod(y.shape))


print y.shape, Y_test.shape
print y[0], Y_test[0]

y1 = np.where(y < 0.5, 0, 1)

#cmap = cm.rainbow(np.linspace(0, 1, 10))
#print type(cmap); quit()
#print colors[3]
#print colors[4];quit()

X = X_test.T
# Need colorto depend on y (two values)
#plt.scatter(X[0],X[1], c=y1)

patches = []

circle = mpatches.Circle((0.,0.), 1.)
patches.append(circle)
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)

xlim = (-3, 3)
ylim = (-3, 3)

X = X_train.T
# Min/max of rainbow are -3 and 2. The values of Y_train are within this range
n = Normalize(vmin=-3.0, vmax=2.)
ax = plt.subplot(1,2,1)
ax.add_collection(collection)
ax.set_xlim((-3,3))
ax.set_ylim((-3,3))
print Y_train
plt.scatter(X[0],X[1], norm=n, c=Y_train, s=50, cmap=cm.rainbow)  # increase scale: s=50
ax = plt.subplot(1,2,2)

# cannot use a collection on two different axes!
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
ax.add_collection(collection)
ax.set_xlim((-3,3))
ax.set_ylim((-3,3))
X = X_test.T
print Y_test
print y1
plt.scatter(X[0],X[1], norm=n, c=y1, s=50, cmap=cm.rainbow)  # increase scale: s=50
plt.show()
quit()

#--------------------------------------------------------

# EXTRACT x where label = 0 and where label = 1 and do 2 scatter plots
# Make colors more visible without increasing the radius
# Create a plotting routine
# Use Tensorboard to plot loss, etc. as a function of time
x1 = X_train[np.equal(Y_train, 0)]
x2 = X_train[np.equal(Y_train, 1)]
print len(x1), len(x2)

for i in xrange(len(y)):
	print y[i], y1[i], Y_test[i]

print "y1-Y_test= ", (y1 - Y_test)
print "nb errors: ", (np.abs(y1 - Y_test)).sum()
# if y[i] < 0.5 ==> 0
# if y[i] >= 0.5 ==> 1

# Notes:
# Dropout of 0.5 seems optimum

#------------------------------------
# I would like to plot the labels
# I would like to run the testing at every epoch (using callbacks?)
