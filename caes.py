from __future__ import print_function

# Python common stuff
import argparse
import sys
try:
    import cPickle as pickle
except:
    import pickle

import scipy.io as sio
import numpy as np

from nolearn.lasagne import NeuralNet

def import_model(model_name):
	global model
	model = __import__(model_name, globals(), locals())

def create_caes(X, model_name):
	import_model(model_name)

	return NeuralNet(
	    layers = model.get_layers(X, 3),
	    max_epochs=10,

	    update=nesterov_momentum,
	    update_learning_rate=0.01,
	    update_momentum=0.975,

	    regression=True,
	    verbose=1
	)

def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Simple convolutional autoencoder')
	parser.add_argument('model_name', metavar = 'model_name', type = str,
		#nargs = '+',
		help = 'Which network parameters should we use?')

	parser.add_argument('--n_epochs', default = 20,
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

	parser.add_argument('--resize_height', default = 1000,
			metavar = 'resize_height', type = int,
			help = 'Resize images to which height?')

	parser.add_argument('--resize_width', default = 750,
			metavar = 'resize_width', type = int,
			help = 'Resize images to which width?')

	return parser.parse_args()

def print_net(ae):
	pass

def train(ae):
	for e in n_epochs:
		for b in load_data():
			# XXX: Do I need to create a copy of `b`?
			ae.partial_fit(b, b)

checkpoint_base_path = 'data/checkpoints/'
results_base_path = 'data/results/'

def main():
	args = parse_command_line()
	model_name = args.model_name
	n_epochs = args.n_epochs
	batch_size = args.batch_size
	checkpoint_every = args.checkpoint_every
	print_every = args.print_every
	resize_height = resize_height
	resize_width = resize_width

	checkpoint_dir = checkpoint_base_path + model_name
	results_dir = results_base_path + model_name

	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	ae = create_caes([batch_size, 3, resize_width, resize_height], model_name)
	train(ae)

	W1 = ae.layers_[1].W.get_value()
	b1 = ae.layers_[1].b.get_value()

	W2 = ae.layers_[3].W.get_value()
	b2 = ae.layers_[3].b.get_value()

if __name__ == '__main__':
	main()

