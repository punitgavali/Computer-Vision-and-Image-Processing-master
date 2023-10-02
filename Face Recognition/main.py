from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
import re
import random
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import os
from os import listdir
from os.path import isfile, join
from glob import glob
import cv2
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

train_dir = 'orl_faces\\'
num_classes = 40
IMG_WIDTH = 92
IMG_HEIGHT = 112
batch_size = 128
epochs = 500

training_data = []

def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
	margin = 1
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
	pairs = []
	labels = []
	n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
	for d in range(num_classes):
		for i in range(n):
			z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
			pairs += [[x[z1], x[z2]]]
			inc = random.randrange(1, num_classes)
			dn = (d + inc) % num_classes
			z1, z2 = digit_indices[d][i], digit_indices[dn][i]
			pairs += [[x[z1], x[z2]]]
			labels += [1, 0]
	return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):
	pred = y_pred.ravel() < 0.5
	return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_train_data():
	face_folders = glob(train_dir+"/*/")    #Going into the training folder and getting sub-directories i.e each of the person's folder

	for i in range(0, len(face_folders)):
		onlyfiles = [f for f in listdir(face_folders[i]) if isfile(join(face_folders[i], f))]  #This contains the path to all the face images of a single person
		face_number = face_folders[i]
		class_num = re.findall(r'\d+',face_number)
		class_num = int(class_num[0])-1
		for j in range(0, len(onlyfiles)):
			img_array = cv2.imread(os.path.join(face_folders[i],onlyfiles[j]), 0)  # We're converting read images into grayscale image since color is not essential for this classification task
			new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))  #Normalizing or Resizing the dataset so that each image is of same size
			training_data.append([new_array, class_num]) 


def reshape_train_data():
	x_train = []
	y_train = []

	for image, label in training_data:
	    x_train.append(image)
	    y_train.append(label)
	    
	x_train = np.array(x_train)
	x_train = np.array(x_train).reshape(x_train.shape[0], IMG_WIDTH, IMG_HEIGHT)

	y_train = np.array(y_train)
	y_train = np.array(y_train).reshape(y_train.shape[0])

	return (x_train, y_train)

create_train_data()
(x_train, y_train) = reshape_train_data()


x_train = x_train.astype('float32')
x_train /= 255

#create positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

x_train , x_test , y_train, y_test = train_test_split(tr_pairs, tr_y, test_size = 0.2, random_state= 3)  #Splitting train and test data. test is 20%.

#Defining our model
def get_model(input_shape):
	input_a = Input(shape=input_shape)
	input_b = Input(shape=input_shape)

	model = Sequential()
	model.add(Conv2D(16, kernel_size = [5,5],activation = 'relu', input_shape = input_shape, name="conv_1"))
	model.add(MaxPooling2D(pool_size=3, name="maxpool_1"))
	model.add(Conv2D(32, kernel_size = [5,5],activation = 'relu', input_shape = input_shape, name="conv_2"))
	model.add(MaxPooling2D(pool_size=3, name="maxpool_2"))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))

	processed_a = model(input_a)
	processed_b = model(input_b)

	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	siamese_net = Model([input_a, input_b], distance)
	plot_model(siamese_net, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	return siamese_net

input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)
model = get_model(input_shape)

#Training
#rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer='sgd', metrics=[accuracy])
left_train = np.array(x_train[:,0]).reshape(len(x_train[:,0]),IMG_WIDTH,IMG_HEIGHT,1)
right_train = np.array(x_train[:,1]).reshape(len(x_train[:,1]),IMG_WIDTH,IMG_HEIGHT,1)
left_test = np.array(x_test[:,0]).reshape(len(x_test[:,0]),IMG_WIDTH,IMG_HEIGHT,1)
right_test = np.array(x_test[:,1]).reshape(len(x_test[:,1]),IMG_WIDTH,IMG_HEIGHT,1)

model.fit([left_train, right_train], y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([left_test, right_test], y_test))

# compute final accuracy on training and test sets
y_pred = model.predict([left_train, right_train])
tr_acc = compute_accuracy(y_train, y_pred)
y_pred = model.predict([left_test, right_test])
te_acc = compute_accuracy(y_test, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
