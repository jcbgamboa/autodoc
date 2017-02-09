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

import argparse
import sys
import os

import dataloader as dl

from caes import import_network

models_base_path = 'models/'
results_base_path = 'data/results/'
checkpoint_base_path = 'data/checkpoints/'

def create_cnn(network_module, network_name):
	print ("network_name", network_name)
	return network_module.get_cnn_network(network_name)

def train_cnn(network_module, network_name,
		dataset, checkpoint_every,
		results_file,
		use_mean_image = True,
		dataset_index = None,
		custom_train_file = None,
		custom_validate_file = None,
		run_as_caes = False):
	# ------- Create folder for checkpoints
	checkpoint_dir = os.path.join(checkpoint_base_path, network_name,
							dataset, 'cnn')
	if (dataset_index is not None):
		checkpoint_dir = os.path.join(checkpoint_base_path,
					network_name, dataset,
					'run_' + str(dataset_index), 'cnn')

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	checkpoint_file = os.path.join(checkpoint_dir, 'model.h5')


	# ------- Initialize data loaders
	ds_train = dl.Dataset(dataset,
			use_mean_image = use_mean_image,
			use_custom_train_file = custom_train_file)
	ds_train.model = 'cnn'
	ds_train.mode = 'train'
	n_classes = ds_train.n_target

	ds_val = dl.Dataset(dataset,
			use_mean_image = use_mean_image,
			use_custom_validate_file = custom_validate_file)
	ds_val.model = 'cnn'
	ds_val.mode = 'validate'
	n_classes = ds_val.n_target

	# ------- Initialize the CNN
	cnn = None
	params = None
	if (os.path.exists(checkpoint_file)):
		cnn = load_model(checkpoint_file)
		params = network_module.get_cnn_parameters()
	else:
		if (run_as_caes):
			input_layer, net, params = network_module.get_caes_network()
		else:
			input_layer, net, params = create_cnn(
							network_module,
							network_name)

		# Add the last layer here because this is dependent on the
		# number of classes (that we only know from `ds_train`)
		#
		# We need a Softmax activation here: this is the last layer
		net = Dense(n_classes, activation = 'softmax')(net)
		cnn = Model(input_layer, net)

		optimizer = Adam(lr = params['learning_rate'],
				beta_1 = params['beta1'],
				beta_2 = params['beta2'])

		cnn.compile(optimizer = optimizer,
				loss = 'categorical_crossentropy',
				metrics = ['accuracy'])

	ds_train.batch_size = params['batch_size']

	# I want to test everything in one batch. My maximum validation set has
	# 200 elements. So I am safe to use 256 here =)
	ds_val.batch_size = 256

	# Needed because the shapes are determined by the pretrained CAES net.
	# We follow the `tf` ordering: row x columns x channels
	input_shape = cnn.layers[0].input_shape[1:]
	ds_train.resize = [input_shape[0], input_shape[1]]
	ds_val.resize  = [input_shape[0], input_shape[1]]

	iteration = 1
	patience = 0
	best_accuracy_so_far = 0
	for epoch in range(params['n_epochs']):
		print("Starting epoch {}".format(epoch))
		curr_batch = 1
		for b in ds_train:
			[loss, accuracy] = cnn.train_on_batch(b[0], b[1])

			output = "Iteration: {}, Batch: {}, Loss: {}, Accuracy: {}"
			print(output.format(iteration, curr_batch,
						loss, accuracy))

			if (iteration % checkpoint_every == 0):
				print("Saving checkpoint")
				cnn.save(checkpoint_file)

			curr_batch += 1
			iteration += 1

		# Saves also by the end of an epoch
		print("Saving checkpoint")
		cnn.save(checkpoint_file)

		for val_b in ds_val:
			# XXX: Fix this hack
			# This for is expected to iterate only once!
			[loss, accuracy] = cnn.test_on_batch(val_b[0], val_b[1])
			print(('Validation: BatchSize: {}; Loss: {}; ' +
				'Accuracy: {}').format(
					val_b[0].shape[0], loss, accuracy))

			# Early stopping
			if (accuracy > best_accuracy_so_far):
				best_accuracy_so_far = accuracy
				patience = 0
				print("Saving best model so far")
				cnn.save(results_file)
			else:
				patience += 1
				if (patience == 10):
					return epoch

	cnn.save(results_file)
	return params['n_epochs']


def dump_cnn(cnn, network_name, results_dir, results_file,
		dataset, test_data, testL,
		dataset_index = None):
	cnn.save(results_file)

	output_results(cnn, results_dir, dataset, test_data, testL)


def output_results(cnn, results_dir, dataset, test_data, testL):
	accuracy_file = os.path.join(results_dir, 'accuracy.csv')
	test_metrics_file = os.path.join(results_dir, 'test_metrics.csv')

	pred = cnn.predict(test_data)

	categorical_pred = np.argmax(pred, axis = 1)
	categorical_testL = np.argmax(testL, axis = 1)
	correctly_classified = categorical_pred == categorical_testL

	test_metrics = zip(categorical_pred,
				categorical_testL,
				correctly_classified)

	np.savetxt(test_metrics_file, list(test_metrics),
				delimiter = ',', fmt = '%d')

	accuracy = len(categorical_pred[categorical_pred == categorical_testL])
	print("Accuracy: ", accuracy, "from:", testL.shape[0])

	with open(accuracy_file, 'w') as f:
		f.write(str(accuracy) + ',' + str(testL.shape[0]))


def get_test_data(cnn, dataset, use_mean_image = True, custom_test_file = None):
	# We are supposing the `test` data not to be to large. We will put the
	# entire thing in the memory.

	# Needed for instantiating the `Dataset` object
	input_shape = cnn.layers[0].input_shape[1:]

	ds_test = dl.Dataset(dataset,
			use_mean_image = use_mean_image,
			use_custom_test_file = custom_test_file)
	ds_test.model = 'cnn'
	ds_test.mode = 'test'
	ds_test.resize  = [input_shape[0], input_shape[1]]
	n_classes = ds_test.n_target

	# Probably there is a more efficient way of implementing this
	[test_data, testL] = next(ds_test)
	for b in ds_test:
		test_data = np.vstack((test_data, b[0]))
		testL = np.vstack((testL, b[1]))

	return test_data, testL


def main():
	args = parse_command_line()

	# -------Loads the network module. It has the network parameters
	network_module = import_network(args.network_name)

	custom_test_file = None
	custom_validate_file = None
	custom_train_file = None
	dataset_index = None
	if (args.dataset_index is not None):
		custom_test_file = 'test_' + str(args.dataset_index) + '.txt'
		custom_validate_file = 'validate_' + str(args.dataset_index) + '.txt'
		custom_train_file = 'train_' + str(args.dataset_index) + '.txt'
		dataset_index = args.dataset_index

	# ------- Create folder for results
	results_dir = os.path.join(results_base_path, args.network_name,
						args.dataset, 'cnn')
	if (dataset_index is not None):
		results_dir = os.path.join(results_base_path, args.network_name,
			args.dataset, 'run_' + str(args.dataset_index), 'cnn')

	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	results_file = os.path.join(results_dir, 'model.h5')


	# ------- Train the CNN
	n_training_epochs = train_cnn(network_module, args.network_name,
			args.dataset, args.checkpoint_every,
			results_file,
			use_mean_image = args.use_mean_image,
			dataset_index = dataset_index,
			custom_validate_file = custom_validate_file,
			custom_train_file = custom_train_file,
			run_as_caes = args.run_as_caes)

	print("Trained for {} epochs.".format(n_training_epochs))

	cnn = load_model(results_file)

	# ------- Dump results
	test_data, testL = get_test_data(cnn, args.dataset,
				use_mean_image = args.use_mean_image,
				custom_test_file = custom_test_file)

	dump_cnn(cnn, args.network_name, results_dir, results_file,
			args.dataset, test_data,
			testL, args.dataset_index)


def parse_command_line():
	description = 'CNN constructed based on a CAES.'
	parser = argparse.ArgumentParser(description = description)

	parser.add_argument('network_name', metavar = 'network_name', type = str,
			help = 'Which network parameters should we use?')

	parser.add_argument('--checkpoint_every', default = 500,
			metavar = 'checkpoint_every', type = int,
			help = 'Save checkpoint after every how many iterations?')

	parser.add_argument('--dataset', default = 'tobacco',
			metavar = 'dataset', type = str,
			help = 'Name of the training dataset.')

	parser.add_argument('--dataset_index', default = None,
			metavar = 'dataset_index', type = int,
			help = 'This is used to run the same network over' + \
				'"resamplings" of the dataset')

	parser.add_argument('--use_mean_image', dest='use_mean_image',
			action='store_true')
	parser.set_defaults(use_mean_image = False)

	parser.add_argument('--run_as_caes', dest='run_as_caes',
			action='store_true')
	parser.set_defaults(run_as_caes = False)

	#parser.add_argument('--custom_test_file', default = 'tobacco',
	#		metavar = 'custom_test_file', type = str,
	#		help = 'Instead of `test.txt`, specify a name for' + \
	#			'the test file to be used.')

	#parser.add_argument('--custom_train_file', default = 'tobacco',
	#		metavar = 'custom_train_file', type = str,
	#		help = 'Instead of `train.txt`, specify a name for' + \
	#			'the test file to be used.')

	#parser.add_argument('--show_network', dest='show_network',
	#		action='store_true')
	#parser.set_defaults(show_network=False)

	return parser.parse_args()

if __name__ == '__main__':
	main()

