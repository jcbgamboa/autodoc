from __future__ import print_function

# Python common stuff
import argparse
import sys
import os
import importlib
import dataloader as dl

import models

try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as sio
import numpy as np

from lasagne.updates import nesterov_momentum, adam

from nolearn.lasagne import NeuralNet, TrainSplit

checkpoint_base_path = 'data/checkpoints/'
results_base_path = 'data/results/'
models_base_path = 'models.'

def import_model(model_name):
	global model
	model_full_path = models_base_path + model_name
	print (model_full_path)
	#model = __import__(model_full_path, globals(), locals())
	model = importlib.import_module(model_full_path)

def create_caes(X, model_name, learning_rate, beta1, beta2, n_epochs = 10):
	import_model(model_name)

	return NeuralNet(
	    layers = model.get_layers(X),
	    max_epochs = n_epochs,

	    update = adam,
	    update_learning_rate = learning_rate,
	    update_beta1 = beta1,
	    update_beta2 = beta2,

	    regression = True,
	    verbose = 1,
	    train_split = TrainSplit(0),
	)

def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Simple convolutional autoencoder')
	parser.add_argument('model_name', metavar = 'model_name', type = str,
		#nargs = '+',
		help = 'Which network parameters should we use?')

	parser.add_argument('--n_epochs', default = 1,
			metavar = 'n_epochs', type = int,
			help = 'Number of epochs through the training set.')

	parser.add_argument('--batch_size', default = 64,
			metavar = 'batch_size', type = int,
			help = 'Size of the training batch.')

	parser.add_argument('--checkpoint_every', default = 500,
			metavar = 'checkpoint_every', type = int,
			help = 'Save checkpoint after every how many iterations?')

	parser.add_argument('--print_every', default = 500,
			metavar = 'print_every', type = int,
			help = 'Print status every how many iterations?')

	parser.add_argument('--resize_height', default = 244,
			metavar = 'resize_height', type = int,
			help = 'Resize images to which height?')

	parser.add_argument('--resize_width', default = 244,
			metavar = 'resize_width', type = int,
			help = 'Resize images to which width?')

	parser.add_argument('--learning_rate', default = 0.00001,
			metavar = 'learning_rate', type = float,
			help = "Adam's learning rate.")

	parser.add_argument('--beta1', default = 0.9,
			metavar = 'beta1', type = float,
			help = "Adam's second momentum decay.")

	parser.add_argument('--beta2', default = 0.999,
			metavar = 'beta2', type = float,
			help = "Adam's first momentum decay.")

	parser.add_argument('--dataset', default = 'tobacco',
			metavar = 'dataset', type = str,
			help = 'Name of the training dataset.')


	return parser.parse_args()

def print_net(ae):
	pass

def train(ae, X, print_every, checkpoint_every, checkpoint_dir,
		n_epochs, dataset):
	checkpoint_file = checkpoint_dir + '/chkpnt.pickle'
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	ds = dl.Dataset(dataset)

	current_iteration = 0
	for e in range(n_epochs):
		print("Starting epoch {}".format(e))
		for b in ds.load_data(batch_size=X[0], resize=[X[2], X[3]]):
			# XXX: Do I need to create a copy of `b`?
			b_out = b[0].reshape((b[0].shape[0], -1))

			ae.partial_fit(b[0], b_out)
			loss = ae.train_history_

			if (current_iteration % checkpoint_every):
				with open(checkpoint_file, 'wb') as f:
					pickle.dump(ae, f, -1)

			if (current_iteration % print_every):
				#print("Iteration {}, loss: {}".format(
				#	current_iteration,
				#	loss))
				print_net(ae)
			current_iteration += 1


def main():
	args = parse_command_line()
	model_name = args.model_name
	n_epochs = args.n_epochs
	batch_size = args.batch_size
	checkpoint_every = args.checkpoint_every
	print_every = args.print_every
	resize_height = args.resize_height
	resize_width = args.resize_width
	learning_rate = args.learning_rate
	dataset = args.dataset
	beta1 = args.beta1
	beta2 = args.beta2

	checkpoint_dir = checkpoint_base_path + model_name
	results_dir = results_base_path + model_name

	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	X = [batch_size, 1, resize_width, resize_height]
	ae = create_caes(X, model_name, learning_rate, beta1, beta2, n_epochs)
	train(ae, X, print_every, checkpoint_every, checkpoint_dir,
			n_epochs, dataset)

	weights = models.get_weights(ae)

if __name__ == '__main__':
	main()

