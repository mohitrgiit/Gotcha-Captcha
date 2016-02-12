import sys,os
from PIL import Image, ImageFilter
import numpy as np
import h5py
import gc

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization

def retrieve_hdf5():
	with h5py.File('training.h5','r') as hf:
		print 'Retrieving hdf5 data...'
		nb_classes = 11

		y1_train = np.array(hf.get('y1_train'))
		Y1_train = np_utils.to_categorical(y1_train, nb_classes)
		del y1_train

		y2_train = np.array(hf.get('y2_train'))
		Y2_train = np_utils.to_categorical(y2_train, nb_classes)
		del y2_train

		y3_train = np.array(hf.get('y3_train'))
		Y3_train = np_utils.to_categorical(y3_train, nb_classes)
		del y3_train

		y4_train = np.array(hf.get('y4_train'))
		Y4_train = np_utils.to_categorical(y4_train, nb_classes)
		del y4_train

		y5_train = np.array(hf.get('y5_train'))
		Y5_train = np_utils.to_categorical(y5_train, nb_classes)
		del y5_train

		print 'Training output data fetched'

		y1_test = np.array(hf.get('y1_test'))
		Y1_test = np_utils.to_categorical(y1_test, nb_classes)
		del y1_test

		y2_test = np.array(hf.get('y2_test'))
		Y2_test = np_utils.to_categorical(y2_test, nb_classes)
		del y2_test

		y3_test = np.array(hf.get('y3_test'))
		Y3_test = np_utils.to_categorical(y3_test, nb_classes)
		del y3_test

		y4_test = np.array(hf.get('y4_test'))
		Y4_test = np_utils.to_categorical(y4_test, nb_classes)
		del y4_test

		y5_test = np.array(hf.get('y5_test'))
		Y5_test = np_utils.to_categorical(y5_test, nb_classes)
		del y5_test

		print 'Testing output data fetched'

		X_train = np.array(hf.get('X_train'))
		print 'Training input data fetched'

		X_test = np.array(hf.get('X_test'))
		print 'Testing input data fetched'

		return (X_train,Y1_train,Y2_train,Y3_train,Y4_train,Y5_train,X_test,Y1_test,Y2_test,Y3_test,Y4_test,Y5_test)

def CNN(X_train,Y1_train,Y2_train,Y3_train,Y4_train,Y5_train,X_test,Y1_test,Y2_test,Y3_test,Y4_test,Y5_test):

	batch_size = 32
	nb_classes = 11
	nb_epoch = 200
	img_rows, img_cols = 50, 50
	img_channels = 3
	X_test=X_test[0:X_test.shape[0]-3]

	print 'Readying vectors...'
	gc.collect()

	X_train = X_train.astype("float16",copy=False)
	X_test = X_test.astype("float16",copy=False)
	X_train /= 255
	X_test /= 255

	print X_train.shape
	print X_test.shape
	print Y1_train.shape
	print Y1_test.shape

	print 'Preparing architecture...'

	graph = Graph()

	graph.add_input(name='input1',input_shape=(img_channels, img_rows, img_cols))

	graph.add_node(Convolution2D(32, 3, 3, border_mode='same'),name='conv0',input='input1')
	graph.add_node(Activation('relu'),name='act0',input='conv0')
	graph.add_node(Convolution2D(32, 3, 3),name='conv1',input='act0')
	graph.add_node(Activation('relu'),name='act1',input='conv1')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool1',input='act1')
	graph.add_node(Dropout(0.25),name='drop1',input='pool1')

	graph.add_node(Convolution2D(64, 3, 3, border_mode='same'),name='conv2',input='pool1')
	graph.add_node(Activation('relu'),name='act2',input='conv2')
	graph.add_node(Convolution2D(64, 3, 3),name='conv3',input='act2')
	graph.add_node(Activation('relu'),name='act3',input='conv3')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool2',input='act3')
	graph.add_node(Dropout(0.25),name='drop2',input='pool2')

	graph.add_node(Convolution2D(128, 3, 3, border_mode='same'),name='conv4',input='drop2')
	graph.add_node(Activation('relu'),name='act4',input='conv4')
	graph.add_node(Convolution2D(128, 3, 3),name='conv5',input='act4')
	graph.add_node(Activation('relu'),name='act5',input='conv5')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool3',input='act5')
	graph.add_node(Dropout(0.25),name='drop3',input='pool3')

	graph.add_node(Convolution2D(256, 3, 3, border_mode='same'),name='conv6',input='drop3')
	graph.add_node(Activation('relu'),name='act6',input='conv6')
	graph.add_node(Convolution2D(256, 3, 3),name='conv7',input='act6')
	graph.add_node(Activation('relu'),name='act7',input='conv7')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool4',input='act7')
	graph.add_node(Dropout(0.25),name='drop4',input='pool4')

	# graph.add_node(Convolution2D(512, 3, 3, border_mode='same'),name='conv8',input='drop4')
	# graph.add_node(Activation('relu'),name='act8',input='conv8')
	# graph.add_node(Convolution2D(512, 3, 3, border_mode='same'),name='conv9',input='act8')
	# graph.add_node(Activation('relu'),name='act9',input='conv9')
	# graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool5',input='act9')
	# graph.add_node(Dropout(0.25),name='drop5',input='pool5')


	graph.add_node(Flatten(),name='flat',input='drop4')
	graph.add_node(Dense(512),name='fc1',input='flat')
	graph.add_node(Activation('relu'),name='act10',input='fc1')
	graph.add_node(Dropout(0.5),name='drop6',input='act10')
	graph.add_node(Dense(512),name='fc2',input='drop6')
	graph.add_node(Activation('relu'),name='act11',input='fc2')
	graph.add_node(Dropout(0.5),name='drop7',input='act11')

	graph.add_node(Dense(nb_classes),name='out1',input='drop7')
	graph.add_node(Activation('softmax'),name='soft1',input='out1')
	graph.add_output(name='digit1',input='soft1')

	graph.add_node(Dense(nb_classes),name='out2',input='drop7')
	graph.add_node(Activation('softmax'),name='soft2',input='out2')
	graph.add_output(name='digit2',input='soft2')

	graph.add_node(Dense(nb_classes),name='out3',input='drop7')
	graph.add_node(Activation('softmax'),name='soft3',input='out3')
	graph.add_output(name='digit3',input='soft3')

	graph.add_node(Dense(nb_classes),name='out4',input='drop7')
	graph.add_node(Activation('softmax'),name='soft4',input='out4')
	graph.add_output(name='digit4',input='soft4')

	graph.add_node(Dense(nb_classes),name='out5',input='drop7')
	graph.add_node(Activation('softmax'),name='soft5',input='out5')
	graph.add_output(name='digit5',input='soft5')

	print 'Starting with training...'
	gc.collect()
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	graph.compile(loss={'digit1':'categorical_crossentropy','digit2':'categorical_crossentropy','digit3':'categorical_crossentropy','digit4':'categorical_crossentropy','digit5':'categorical_crossentropy'}, optimizer=sgd)
	history = graph.fit({'input1':X_train, 'digit1':Y1_train, 'digit2':Y2_train, 'digit3':Y3_train, 'digit4':Y4_train, 'digit5':Y5_train},nb_epoch=nb_epoch,batch_size=batch_size,verbose=1)
	score = graph.evaluate({'input1':X_test, 'digit1':Y1_test, 'digit2':Y2_test, 'digit3':Y3_test, 'digit4':Y4_test, 'digit5':Y5_test}, batch_size=batch_size)
	print score

	# print("Using real time data augmentation")

	# datagen = ImageDataGenerator(
	# 	featurewise_center=True,  # set input mean to 0 over the dataset
	# 	samplewise_center=False,  # set each sample mean to 0
	# 	featurewise_std_normalization=True,  # divide inputs by std of the dataset
	# 	samplewise_std_normalization=False,  # divide each input by its std
	# 	zca_whitening=False,  # apply ZCA whitening
	# 	rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	# 	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	# 	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	# 	horizontal_flip=True,  # randomly flip images
	# 	vertical_flip=True)  # randomly flip images

	# # compute quantities required for featurewise normalization
	# # (std, mean, and principal components if ZCA whitening is applied)
	# datagen.fit(X_train)

	# graph.fit_generator(datagen.flow(X_train, Y1_train, batch_size=batch_size),
	# 					samples_per_epoch=X_train.shape[0],
	# 					nb_epoch=nb_epoch, verbose=1,
	# 					validation_data=(X_test, Y1_test),
	# 					nb_worker=1)
	# graph.fit_generator(datagen.flow(X_train, Y2_train, batch_size=batch_size),
	# 					samples_per_epoch=X_train.shape[0],
	# 					nb_epoch=nb_epoch, verbose=1,
	# 					validation_data=(X_test, Y2_test),
	# 					nb_worker=1)
	# graph.fit_generator(datagen.flow(X_train, Y3_train, batch_size=batch_size),
	# 					samples_per_epoch=X_train.shape[0],
	# 					nb_epoch=nb_epoch, show_accuracy=True,
	# 					validation_data=(X_test, Y3_test),
	# 					nb_worker=1)
	# graph.fit_generator(datagen.flow(X_train, Y4_train, batch_size=batch_size),
	# 					samples_per_epoch=X_train.shape[0],
	# 					nb_epoch=nb_epoch, verbose=1,
	# 					validation_data=(X_test, Y4_test),
	# 					nb_worker=1)
	# graph.fit_generator(datagen.flow(X_train, Y5_train, batch_size=batch_size),
	# 					samples_per_epoch=X_train.shape[0],
	# 					nb_epoch=nb_epoch, verbose=1,
	# 					validation_data=(X_test, Y5_test),
	# 					nb_worker=1)




data=retrieve_hdf5()
CNN(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11])