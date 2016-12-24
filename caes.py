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

import models.model_1

from nolearn.lasagne import NeuralNet

def create_caes(X, use_rgb):

	return NeuralNet(
	    layers = models.model_1.get_layers(X, 3),
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
	parser.add_argument('in_file', metavar = 'input_data', type = str,
		#nargs = '+',
		help = 'File with the data to be "autoencoded".')
	parser.add_argument('out_file', metavar = 'output_file', type = str,
		#nargs = '+',
		help = 'Name of the output file.')
	parser.add_argument('--use_rgb', dest = 'use_rgb', action='store_true',
		help = 'Use this flag if the input data has three channels.')
	parser.add_argument('--normalize', dest = 'normalize',
		action='store_true',
		help = 'Rescale the data into the interval [0, 1].')

	return parser.parse_args()

def train(ae):
	for e in n_epochs:
		for b in load_data():
			# XXX: Do I need to create a copy of `b`?
			ae.partial_fit(b, b)

def main():
	args = parse_command_line()
	in_file = args.in_file
	out_file = args.out_file
	use_rgb = args.use_rgb

	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	ae = create_caes(X, use_rgb)
	train(ae)
	#ae.fit(X, X_out)

	W1 = ae.layers_[1].W.get_value()
	b1 = ae.layers_[1].b.get_value()

	W2 = ae.layers_[3].W.get_value()
	b2 = ae.layers_[3].b.get_value()

	sio.savemat(out_file, {'caes_W1':W1, 'caes_b1':b1,
					'caes_W2':W2, 'caes_b2':b2})

if __name__ == '__main__':
	main()

