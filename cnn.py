# Generate a simple CNN with Keras

# It receives as parameter
# * a .mat file with the images to be run
# * and optionally another .mat file with weights

# It runs the CNN and spits
# 1) The loss value for each iteration
# 2) The accuracy on the `test set`

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, \
			Activation, Flatten
from keras.callbacks import TensorBoard, Callback

import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp

import matplotlib.pyplot as plt

import argparse
import sys

# For "math.inf", i.e., infinite
import math

def load_model(pretrained_model_file):
	premod = sio.loadmat(pretrained_model_file)

	# (I was dumb and didn't make a homogeneous structure for both nets)
	if 'cdbn_out' in premod.keys():
		W1 = premod['cdbn_out'][0][0]
		b1 = premod['cdbn_out'][0][1].squeeze()
		W2 = premod['cdbn_out'][1][0]
		b2 = premod['cdbn_out'][1][1].squeeze()
	elif 'caes_W1' in premod.keys():
		W1 = np.transpose(premod['caes_W1'], axes = [2, 3, 1, 0])
		W2 = np.transpose(premod['caes_W2'], axes = [2, 3, 1, 0])
		b1 = premod['caes_b1'].squeeze()
		b2 = premod['caes_b2'].squeeze()
	else:
		print("WARN: unrecognized pretrained model format.")
		return None

	print('W1.shape: {}'.format(W1.shape))

	ret = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
	return ret

def create_cnn(pretrained_model_file = None,
		data_shape = (60000, 28, 28, 1),
		activation = 'sigmoid'):
	# Loads weights from other models
	pm = None
	if(pretrained_model_file is not None):
		pm = load_model(pretrained_model_file)

	input_img = Input(shape=(data_shape[1], data_shape[2], data_shape[3]))
	# Defines the CNN architecture (based on the weights?)
	x = Convolution2D(9, 7, 7, border_mode = 'valid',
		weights = None if pm is None else (pm['W1'], pm['b1']),
				activation = activation)(input_img)
	x = MaxPooling2D((2, 2), border_mode = 'same')(x)

	x = Convolution2D(16, 6, 6, border_mode = 'valid',
		weights = None if pm is None else (pm['W2'], pm['b2']),
				activation = activation)(x)
	x = MaxPooling2D((2, 2), border_mode = 'same')(x)

	x = Flatten()(x)
	x = Dense(10, activation = activation)(x)
	x = Activation('softmax')(x)

	#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer=sgd)u

	return Model(input = input_img, output = x)

def label_binarize(trainL, testL):
	binarizer = skp.LabelBinarizer(0, 1, False)
	binarizer.fit(trainL)
	trainL = binarizer.transform(trainL)
	testL = binarizer.transform(testL)
	return trainL, testL

# Strong based on https://keras.io/callbacks/#example-recording-loss-history
class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []
		self.one_minus_accuracy = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracy.append(logs.get('acc'))
		self.one_minus_accuracy.append(1 - logs.get('acc'))

def load_dataset(in_file, normalize = False, reduce_dataset_to = 0):
	dataset = sio.loadmat(in_file)

	if (reduce_dataset_to > dataset['train_data'].shape[3]):
		print("reduce_dataset_to's value is bigger than dataset size")
		sys.exit()

	train_data = dataset['train_data']
	trainL = dataset['trainL']
	testL = dataset['testL']
	if (reduce_dataset_to > 0):
		train_data = train_data[:, :, :, 0:reduce_dataset_to]
		trainL = trainL[0:reduce_dataset_to]

	# Transposing stuff only because I am using Keras+Tensorflow
	train_data = np.transpose(train_data, axes = (3, 0, 1, 2))
	test_data  = np.transpose(dataset['test_data'], axes = (3, 0, 1, 2))

	if (normalize):
		train_data = train_data / 256.0
		test_data  = test_data  / 256.0

	trainL, testL = label_binarize(trainL, testL)

	return (train_data, trainL, test_data, testL)

def output_results(pred, testL, batch_loss_hist, h, out_folder):
	per_batch_metrics = zip(batch_loss_hist.losses,
				batch_loss_hist.accuracy,
				batch_loss_hist.one_minus_accuracy)
	np.savetxt(out_folder + '/per_batch_metrics.csv',
			list(per_batch_metrics), delimiter = ',', fmt = '%f')

	categorical_pred = np.argmax(pred, axis = 1)
	categorical_testL = np.argmax(testL, axis = 1)
	correctly_classified = categorical_pred == categorical_testL

	test_metrics = zip(categorical_pred, categorical_testL,
				correctly_classified)
	np.savetxt(out_folder + '/test_metrics.csv', list(test_metrics),
						delimiter = ',', fmt = '%d')

	accuracy = len(categorical_pred[categorical_pred == categorical_testL])
	# numpy's savetxt() complains if the array has only one element
	with open(out_folder + '/accuracy.csv', 'w') as f:
		f.write(str(accuracy))

	per_epoch_metrics = zip(h.history['loss'], h.history['acc'],
				h.history['categorical_crossentropy'])
	np.savetxt(out_folder + '/per_epoch_metrics.csv',
			list(per_epoch_metrics), delimiter = ',', fmt = '%f')

	return accuracy

def main():
	args = parse_command_line()
	pretrained_model_file = args.pretrained_model
	in_file = args.in_file
	out_folder = args.out_folder
	reduce_dataset_to = args.reduce_dataset_to
	normalize = args.normalize
	use_relu = args.use_relu
	n_epochs = args.n_epochs

	(train_data, trainL, test_data, testL) = load_dataset(in_file,
						normalize, reduce_dataset_to)

	activation = 'sigmoid' if not use_relu else 'relu'
	model = create_cnn(pretrained_model_file, train_data.shape, activation)
	model.compile(optimizer = 'adam',
			loss = 'categorical_crossentropy',
			metrics = ['accuracy', 'categorical_crossentropy'])

	batch_loss_hist = LossHistory()
	h = model.fit(train_data, trainL,
		nb_epoch = n_epochs,
		batch_size = 64,
		callbacks = [TensorBoard(log_dir='./caes_keras_results'),
				batch_loss_hist])

	# Now I want to record the loss on the test set
	pred = model.predict(test_data)

	accuracy = output_results(pred, testL, batch_loss_hist, h, out_folder)
	print("accuracy: {}".format(accuracy))
	#print(batch_loss_hist.losses)
	#plt.plot(batch_loss_hist.losses)
	#plt.show()


def parse_command_line():
	# TODO: add better description
	description = 'Simple CNN.'
	parser = argparse.ArgumentParser(description = description)
	parser.add_argument('--pretrained_model',
			metavar = 'pretrained_model', type = str,
			help = 'Pretrained weights to be used in CNN. ' +
				'Random initialization is used if left empty.')
	parser.add_argument('in_file', metavar = 'in_file', type = str,
			help = 'File with the dataset to be learnt.')
	parser.add_argument('out_folder', metavar = 'output_folder', type = str,
			help = 'Folder where results should be put.')
	parser.add_argument('--reduce_dataset_to', default = 0,
			metavar = 'reduce_dataset_to', type = int,
			help = 'Reduces the size of the dataset to the number' +
				'passed as parameter.')
	parser.add_argument('--normalize', dest = 'normalize',
			action='store_true',
			help = 'Rescale the data into the interval [0, 1].')
	parser.add_argument('--use_relu', dest = 'use_relu',
			action='store_true',
			help = 'Use ReLU activations instead of sigmoid.')
	parser.add_argument('--n_epochs', default = 1,
			metavar = 'n_epochs', type = int,
			help = 'Number of epochs through the training set.')

	return parser.parse_args()

if __name__ == '__main__':
	main()

