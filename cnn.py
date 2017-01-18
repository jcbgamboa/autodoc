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

def create_cnn(network_module, network_name):
	return network_module.get_cnn_network(network_name)

def train_cnn(network_module, network_name, dataset, checkpoint_every):
	checkpoint_dir = os.path.join(checkpoint_base_path, network_name,
							dataset, 'cnn')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_file = os.path.join(checkpoint_dir, 'model.h5')

	ds_train = dl.Dataset(dataset)
	ds_train.model = 'cnn'
	ds_train.mode = 'train'
	n_classes = ds_train.n_target

	ds_test = dl.Dataset(dataset)
	ds_test.model = 'cnn'
	ds_test.mode = 'validate'
	n_classes = ds_test.n_target

	cnn = None
	params = None
	if (os.path.exists(checkpoint_file)):
		cnn = load_model(checkpoint_file)
		params = network_module.get_cnn_parameters()
	else:
		input_layer, net, params = create_cnn(network_module,
							network_name)

		# Add the last layer here because this is dependent on the
		# number of classes (that we only know from `ds_train`)
		net = Dense(n_classes, activation=params['activation'])(net)
		cnn = Model(input_layer, net)

		optimizer = Adam(lr = params['learning_rate'],
				beta_1 = params['beta1'],
				beta_2 = params['beta2'])

		cnn.compile(optimizer = optimizer,
				loss = 'mean_squared_error',
				metrics = ['accuracy'])


	ds_train.batch_size = params['batch_size']
	ds_test.batch_size = params['batch_size']

	# Needed because the shapes are determined by the pretrained CAES net.
	# We follow the `tf` ordering: row x columns x channels
	input_shape = cnn.layers[0].input_shape[1:]
	ds_train.resize = [input_shape[0], input_shape[1]]
	ds_test.resize  = [input_shape[0], input_shape[1]]

	val_b = next(ds_test)

	iteration = 1
	for epoch in range(params['n_epochs']):
		print("Starting epoch {}".format(epoch))
		curr_batch = 1
		for b in ds_train:
			[loss, accuracy] = cnn.train_on_batch(b[0], b[1])

			output = "Iteration: {}, Batch: {}, Loss: {}, Accuracy: {}"
			print(output.format(iteration, curr_batch,
						loss, accuracy))

			if (iteration % checkpoint_every == 0):
				print("Saving model")
				cnn.save(checkpoint_file)

			curr_batch += 1
			iteration += 1

		# Saves also by the end of an epoch
		print("Saving model")
		cnn.save(checkpoint_file)

		#for b in ds_test:
		[loss, accuracy] = cnn.test_on_batch(val_b[0], val_b[1])
		print('Validation Loss: {}; Accuracy: {}'.format(
						loss, accuracy))
		#	print('validate loss: ' + str(loss))
		#	print('validate accuracy: ' + str(accuracy))


	return cnn


def dump_cnn(cnn, network_name, dataset):
	results_dir = os.path.join(results_base_path, network_name,
					dataset, 'cnn')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	results_file = os.path.join(results_dir, 'model.h5')
	cnn.save(results_file)


def main():
	args = parse_command_line()

	#show_network = args.show_network
	#if (show_network):
	#	show_caes_network(network_name)
	#	sys.exit()

	# Loads the network module. It has the network parameters
	network_module = import_network(args.network_name)

	cnn = train_cnn(network_module, args.network_name,
			args.dataset, args.checkpoint_every)

	dump_cnn(cnn, args.network_name, args.dataset)


def parse_command_line():
	# TODO: add better description
	description = 'Simple CNN.'
	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('network_name', metavar = 'network_name', type = str,
			help = 'Which network parameters should we use?')

	parser.add_argument('--checkpoint_every', default = 500,
			metavar = 'checkpoint_every', type = int,
			help = 'Save checkpoint after every how many iterations?')

	parser.add_argument('--dataset', default = 'tobacco',
			metavar = 'dataset', type = str,
			help = 'Name of the training dataset.')

	parser.add_argument('--show_network', dest='show_network',
			action='store_true')
	parser.set_defaults(show_network=False)


	return parser.parse_args()

if __name__ == '__main__':
	main()

