import argparse
import importlib

import sys
import os

import dataloader as dl

from keras.models import Model, load_model

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

def train_caes(network_module, network_name, dataset):
	checkpoint_dir = os.path.join(checkpoint_base_path, network_name,
					dataset, 'caes')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_file = os.path.join(checkpoint_dir, 'chkpnt.pickle')

	caes = None
	param = None
	if (os.path.exists(checkpoint_file)):
		caes = load_model(checkpoint_file)
	else:
		input_layer, net, params = create_caes(network_module)

		caes = Model(input_layer, net)
		caes.compile(optimizer = 'adam',
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

	for epoch in range(params['n_epochs']):
		print("Starting epoch {}".format(epoch))
		for b in ds_train:
			print('ds_train.gen_counter', ds_train.gen_counter)
			#b_out = b[0].reshape((b[0].shape[0], -1))
			[loss, accuracy] = caes.train_on_batch(b[0], b[0])
			print("Trained a batch. Loss: {}, Accuracy: {}".format(
				loss, accuracy))

		# For now, saving every epoch
		caes.save(os.path.join(checkpoint_dir, 'model.h5'))

		for b in ds_test:
			#b_out = b[0].reshape((b[0].shape[0], -1))
			[loss, accuracy] = caes.test_on_batch(b[0], b[0])

			print('validate loss: ' + str(loss))
			print('validate accuracy: ' + str(accuracy))
		


def dump_caes(caes):
	results_dir = os.path.join(results_base_path, model_name,
					dataset, 'caes')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	results_file = os.path.join(results_dir, 'model.pickle')

	caes.save(results_file)


def main():
	args = parse_command_line()

	show_network = args.show_network
	if (show_network):
		show_caes_network(args.network_name)
		sys.exit()

	# Loads the network module. It has the network parameters
	network_module = import_network(args.network_name)

	train_caes(network_module, args.network_name, args.dataset)

	dump_caes(caes)


def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Simple convolutional autoencoder')
	parser.add_argument('network_name', metavar = 'network_name', type = str,
		#nargs = '+',
		help = 'Which network parameters should we use?')

	parser.add_argument('--checkpoint_every', default = 500,
			metavar = 'checkpoint_every', type = int,
			help = 'Save checkpoint after every how many iterations?')

	parser.add_argument('--print_every', default = 500,
			metavar = 'print_every', type = int,
			help = 'Print status every how many iterations?')

	parser.add_argument('--dataset', default = 'tobacco',
			metavar = 'dataset', type = str,
			help = 'Name of the training dataset.')

	parser.add_argument('--show_network', dest='show_network',
						action='store_true')
	parser.set_defaults(show_network=False)

	return parser.parse_args()



if __name__ == '__main__':
	main()


