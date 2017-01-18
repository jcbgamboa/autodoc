import argparse
import importlib

import sys
import os

import dataloader as dl

from keras.models import Model, load_model
from keras.optimizers import Adam

checkpoint_base_path = 'data/checkpoints/'
results_base_path = 'data/results/'
networks_base_path = 'networks.'

def import_network(network_name):
	global network_module
	network_full_path = networks_base_path + network_name
	print (network_full_path)
	#network = __import__(network_full_path, globals(), locals())
	network_module = importlib.import_module(network_full_path)
	return network_module

def show_caes_network(network_name):
	print("For now... shows nothing")

def create_caes(network_module):
	return network_module.get_caes_network()

def train_caes(network_module, network_name, dataset, checkpoint_every):
	checkpoint_dir = os.path.join(checkpoint_base_path, network_name,
					dataset, 'caes')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_file = os.path.join(checkpoint_dir, 'model.h5')

	caes = None
	params = None
	if (os.path.exists(checkpoint_file)):
		caes = load_model(checkpoint_file)
		params = network_module.get_caes_parameters()
	else:
		input_layer, net, params = create_caes(network_module)

		caes = Model(input_layer, net)

		optimizer = Adam(lr = params['learning_rate'],
				beta_1 = params['beta1'],
				beta_2 = params['beta2'])

		caes.compile(optimizer = optimizer,
			loss = 'mean_squared_error',
			metrics = ['accuracy'])

	ds_train = dl.Dataset(dataset)
	ds_train.resize = [params['rows'], params['columns']]
	ds_train.batch_size = params['batch_size']
	ds_train.model = 'caes'
	ds_train.mode = 'train'

	ds_test = dl.Dataset(dataset)
	ds_test.resize = [params['rows'], params['columns']]
	ds_test.batch_size = params['batch_size']
	ds_test.model = 'caes'
	ds_test.mode = 'validate'

	val_b = next(ds_test)

	iteration = 1
	for epoch in range(params['n_epochs']):
		print("Starting epoch {}".format(epoch))
		curr_batch = 1
		for b in ds_train:
			[loss, accuracy] = caes.train_on_batch(b[0], b[0])

			output = "Iteration: {}, Batch: {}, Loss: {}, Accuracy: {}"
			print(output.format(iteration, curr_batch,
						loss, accuracy))

			if (iteration % checkpoint_every == 0):
				print("Saving model")
				caes.save(checkpoint_file)

			curr_batch += 1
			iteration += 1

		# Saves also by the end of an epoch
		caes.save(checkpoint_file)

		# Run on only one batch, to have an idea of how it is
		[loss, accuracy] = caes.test_on_batch(val_b[0], val_b[0])
		#for b in ds_test:
		#	[loss, accuracy] = caes.test_on_batch(b[0], b[0])
		print('Validation Loss: {}; Accuracy: {}'.format(
						loss, accuracy))

	return caes

def dump_caes(caes, network_name, dataset):
	results_dir = os.path.join(results_base_path, network_name,
					dataset, 'caes')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	results_file = os.path.join(results_dir, 'model.h5')
	caes.save(results_file)


def main():
	args = parse_command_line()

	show_network = args.show_network
	if (show_network):
		show_caes_network(args.network_name)
		sys.exit()

	# Loads the network module. It has the network parameters
	network_module = import_network(args.network_name)

	caes = train_caes(network_module, args.network_name,
			args.dataset, args.checkpoint_every)

	dump_caes(caes, args.network_name, args.dataset)


def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Simple convolutional autoencoder')
	parser.add_argument('network_name', metavar = 'network_name', type = str,
		#nargs = '+',
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


