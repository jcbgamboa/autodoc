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
import pickle

import matplotlib.pyplot as plt

import argparse
import sys
import os
import time

# For "math.inf", i.e., infinite
import math

from sklearn.datasets.tests.test_svmlight_format import currdir

import dataloader as dl

from caes import import_model
import models

models_base_path = 'models/'
results_base_path = 'data/results/'
checkpoint_base_path = 'data/checkpoints/'


def load_caes_model(model_name):
	# Loads weights from other models
	if(model_name is None):
		raise ValueError('Pretrained model not passed.')

	model = import_model(model_name)
	pretrained_model_file = os.path.join(results_base_path,
						model_name,
						'caes/model.pickle')

	ae = pickle.load(open(pretrained_model_file, "rb"))
	return model, ae


def insert_econv2D(model, layer, weights = None):
	activation = 'sigmoid'
	if (layer[1]['nonlinearity'] == model.rectify):
		activation = 'relu'

	return Convolution2D(layer[1]['num_filters'],
			layer[1]['filter_size'],
			layer[1]['filter_size'],
			border_mode = layer[1]['pad'],
			weights = (np.transpose(weights[0], axes = [2, 3, 1, 0]),
					   weights[1]),
			activation = activation)


def insert_maxpool2D(model, layer, weights = None):
	return MaxPooling2D((layer[1]['pool_size'], layer[1]['pool_size']),
			border_mode = 'same')


def insert_batchnorm(model, layer, weights = None):
	return BatchNormalization()


def nolearn_convert_to_keras(nolearn_model, X, model, weights,
				n_classes, activation,
				learning_rate, beta1, beta2):

	print("X: {}".format(X))
	sys.exit()

	switcher = {
		'e_conv2D'  :  insert_econv2D,
		'maxpool2D' :  insert_maxpool2D,
		#'reshape'   :  insert_reshape,ol
		'batchnorm' :  insert_batchnorm,
	}

	net = []
	input_net = []
	curr_w = 0
	for i, l in enumerate(nolearn_model):
		if (l[1]['name'].startswith('mid_')):
			break

		if (l[1]['name'] == 'input'):
			if (i != 0):
				raise Exception("The first layer should be "
						"an input layer.")
			net = input_net = Input(shape = (X[2], X[3], X[1]))
			continue

		for key in switcher.keys():
			if (key in l[1]['name']):

				w = None
				if curr_w < len(weights):
					w = weights[curr_w]

				net = switcher[key](model, l, w)(net)
				if key.startswith('e_conv'):
					curr_w += 1
				break

	net = Flatten()(net)
	net = Dense(512, activation = activation)(net)
	net = Dropout(0.5)(net)
	net = BatchNormalization()(net)
	net = Dense(1024, activation=activation)(net)
	net = Dropout(0.5)(net)
	net = BatchNormalization()(net)
	net = Dense(n_classes, activation=activation)(net)
	#net = Activation('softmax')(net)
	adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2,
			epsilon=1e-08, decay=0.0)

	return Model(input = input_net, output = net), adam


def create_cnn(n_classes,
		model_name = None,
		activation = 'relu',
		learning_rate = 0.00001, beta1 = 0.9, beta2 = 0.999):
	cnn = network_module.get_cnn_network()

	# TODO: load cnn

	return cnn


	model, ae = load_caes_model(model_name)

	input_shape = models.get_input_shape(ae)
	layers = model.get_layers(input_shape)

	(mod, update) = nolearn_convert_to_keras(layers,
					input_shape,
					model,
					models.get_weights(ae),
					n_classes, activation,
					learning_rate,
					beta1, beta2)
	return input_shape, mod, update


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


def show_caes_model(model_name):
	model, ae = load_caes_model(model_name)

	print("ae.update_learning_rate: {}".format(ae.update_learning_rate))
	print("ae.train_split.eval_size: {}".format(ae.train_split.eval_size))
	print("input size: {}".format(models.get_input_shape(ae)))
	print("ae.max_epochs: {}".format(ae.max_epochs))


def main():
	args = parse_command_line()
	model_name = args.model_name
	dataset = args.dataset
	#n_epochs = args.n_epochs
	#batch_size = args.batch_size
	#learning_rate = args.learning_rate
	#beta1 = args.beta1
	#beta2 = args.beta2
	#activation = args.activation

	# Maybe we don't want to run anything. Just show the model
	show_model = args.show_model
	if (show_model):
		show_caes_model(model_name)
		sys.exit()

	ds = dl.Dataset(dataset)
	ds.model = 'cnn'
	n_classes = ds.n_target

	input_shape, model, optimizer = create_cnn(n_classes, model_name, activation,
					learning_rate, beta1, beta2)

	model.compile(optimizer = optimizer,
			loss = 'categorical_crossentropy',
			metrics = ['accuracy', 'categorical_crossentropy'])

	results_dir = os.path.join(results_base_path, model_name,
					dataset, 'cnn')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	checkpoint_dir = os.path.join(checkpoint_base_path, model_name,
					dataset, 'cnn')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	checkpoint_file = os.path.join(checkpoint_dir , 'model.h5')
	checkpointer = ModelCheckpoint(filepath = checkpoint_file,
					verbose = 1, save_best_only = True)

	batch_loss_hist = LossHistory()

	ds.batch_size = batch_size
	ds.resize = [input_shape[2], input_shape[3]]

	if os.path.exists(checkpoint_file):
		model = load_model(checkpoint_file)

	h = model.fit_generator(ds,
		samples_per_epoch = 3482,
		nb_epoch = n_epochs,
		callbacks = [batch_loss_hist])

	time.sleep(10)

	model.save(checkpoint_file)

	test_ds = dl.Dataset('tobacco')
	test_ds.model = 'cnn'
	test_ds.batch_size = batch_size
	test_ds.resize = [input_shape[2], input_shape[3]]

	# Now I want to record the loss on the test set
	for d in test_ds:
		if test_ds.gen_counter > 2000:
			break

		pred = model.predict(d[0], batch_size = batch_size)
		accuracy = output_results(pred, d[1], batch_loss_hist,
						h, results_dir)

	print("accuracy: {}".format(accuracy))
	print(batch_loss_hist.losses)
	#plt.plot(batch_loss_hist.losses)
	#plt.show()


def parse_command_line():
	# TODO: add better description
	description = 'Simple CNN.'
	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('model_name', metavar = 'model_name', type = str,
			help = 'Which network parameters should we use?')

	#parser.add_argument('--n_epochs', default = 1,
	#		metavar = 'n_epochs', type = int,
	#		help = 'Number of epochs through the training set.')

	#parser.add_argument('--batch_size', default = 64,
	#		metavar = 'batch_size', type = int,
	#		help = 'Size of the training batch.')

	#parser.add_argument('--learning_rate', default = 0.00001,
	#		metavar = 'learning_rate', type = float,
	#		help = "Adam's learning rate.")

	#parser.add_argument('--beta1', default = 0.9,
	#		metavar = 'beta1', type = float,
	#		help = "Adam's second momentum decay.")

	#parser.add_argument('--beta2', default = 0.999,
	#		metavar = 'beta2', type = float,
	#		help = "Adam's first momentum decay.")

	parser.add_argument('--dataset', default = 'tobacco',
			metavar = 'dataset', type = str,
			help = 'Name of the training dataset.')

	#parser.add_argument('--activation', default = 'sigmoid',
	#		metavar = 'activation', type = str,
	#		help = 'Activation function of the last layer.')

	parser.add_argument('--show_model', dest='show_model', action='store_true')
	parser.set_defaults(show_model=False)


	return parser.parse_args()

if __name__ == '__main__':
	main()

