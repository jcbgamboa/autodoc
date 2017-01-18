# Generate a simple CNN with Keras

# It receives as parameter
# * a .mat file with the images to be run
# * and optionally another .mat file with weights

# It runs the CNN and spits
# 1) The loss value for each iteration
# 2) The accuracy on the `test set`

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, \
			Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from keras.optimizers import Adam

import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp

import matplotlib.pyplot as plt

import argparse
import sys
import os
import time

# For "math.inf", i.e., infinite
import math

from sklearn.datasets.tests.test_svmlight_format import currdir

import dataloader as dl

from caes import import_network

models_base_path = 'models/'
results_base_path = 'data/results/'
checkpoint_base_path = 'data/checkpoints/'

def create_cnn(network_module):
	return network_module.get_cnn_network(network_name)

def train_cnn(network_module, network_name, dataset):
	checkpoint_dir = os.path.join(checkpoint_base_path, network_name,
					dataset, 'caes')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_file = os.path.join(checkpoint_dir, 'model.h5')

	ds = dl.Dataset(dataset)
	ds.model = 'cnn'
	n_classes = ds.n_target

	caes = None
	params = None
	if (os.path.exists(checkpoint_file)):
		cnn = load_model(checkpoint_file)
	else:
		input_layer, net, params = create_cnn(network_name)

		# Add the last layer, because this is dependent on the number
		# of classes
		net = Dense(n_classes, activation=activation)(net)

		cnn = Model(input_layer, net)

		optimizer = Adam(lr = params['learning_rate'],
				beta_1 = params['beta1'],
				beta_2 = params['beta2'])

		cnn.compile(optimizer = optimizer,
				loss = 'mean_squared_error',
				metrics = ['accuracy'])


	ds.batch_size = params['batch_size']

	# This follows the `tf` ordering: row x columns x channels
	input_shape = cnn.layers[0].input_shape[1:]
	ds.resize = [input_shape[0], input_shape[1]]

	iteration = 1
	for epoch in range(params['n_epochs']):
		print("Starting epoch {}".format(epoch))
		curr_batch = 1
		for b in ds_train:
			[loss, accuracy] = caes.train_on_batch(b[0], ...)

			output = "Iteration: {}, Batch: {}, Loss: {}, Accuracy: {}"
			print(output.format(iteration, curr_batch,
						loss, accuracy))

			if (iteration % checkpoint_every):
				caes.save(os.path.join(checkpoint_dir, 'model.h5'))

			curr_batch += 1
			iteration += 1

		# Saves also by the end of an epoch
		caes.save(checkpoint_file)

		for b in ds_test:
			[loss, accuracy] = caes.test_on_batch(b[0], ...)

			print('validate loss: ' + str(loss))
			print('validate accuracy: ' + str(accuracy))


	return cnn


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

	#show_network = args.show_network
	#if (show_network):
	#	show_caes_model(network_name)
	#	sys.exit()

	# Loads the network module. It has the network parameters
	network_module = import_network(args.network_name)

	cnn = train_cnn(args.network_module, args.network_name, args.dataset)

	dump_cnn(cnn)


def parse_command_line():
	# TODO: add better description
	description = 'Simple CNN.'
	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('model_name', metavar = 'model_name', type = str,
			help = 'Which network parameters should we use?')

	parser.add_argument('--dataset', default = 'tobacco',
			metavar = 'dataset', type = str,
			help = 'Name of the training dataset.')

	parser.add_argument('--show_model', dest='show_model', action='store_true')
	parser.set_defaults(show_model=False)


	return parser.parse_args()

if __name__ == '__main__':
	main()

