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
	return model


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

	#parser.add_argument('--n_epochs', default = 1,
	#		metavar = 'n_epochs', type = int,
	#		help = 'Number of epochs through the training set.')

	#parser.add_argument('--batch_size', default = 64,
	#		metavar = 'batch_size', type = int,
	#		help = 'Size of the training batch.')

	parser.add_argument('--checkpoint_every', default = 500,
			metavar = 'checkpoint_every', type = int,
			help = 'Save checkpoint after every how many iterations?')

	parser.add_argument('--print_every', default = 500,
			metavar = 'print_every', type = int,
			help = 'Print status every how many iterations?')

	#parser.add_argument('--rows', default = 500,
	#		metavar = 'rows', type = int,
	#		help = 'Resize images to which height?')

	#parser.add_argument('--columns', default = 500,
	#		metavar = 'columns', type = int,
	#		help = 'Resize images to which width?')

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

	return parser.parse_args()


def print_net(ae):
	pass


def train(ae, X, print_every, checkpoint_every, checkpoint_dir,
		n_epochs, dataset, model_name):
	checkpoint_file = os.path.join(checkpoint_dir, 'chkpnt.pickle')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	ds = dl.Dataset(dataset)
	ds.model = 'caes'
	current_iteration = 1
	for e in range(n_epochs):
		print("Starting epoch {}".format(e))
		for b in ds.load_data(batch_size=X[0], resize=[X[2], X[3]]):
			# XXX: Do I need to create a copy of `b`?
			b_out = b[0].reshape((b[0].shape[0], -1))

			ae.partial_fit(b[0], b_out)
			loss = ae.train_history_

			if (current_iteration % checkpoint_every == 0):
				with open(checkpoint_file, 'wb') as f:
					pickle.dump(ae, f, -1)

			if (current_iteration % print_every == 0):
				print("Iteration {}, loss: {}".format(
					current_iteration,
					loss))
				print_net(ae)
			current_iteration += 1


def dump_results(ae, results_dir):
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	results_file = os.path.join(results_dir, 'model.pickle')
	with open(results_file, 'wb') as f:
		pickle.dump(ae, f, -1)

	history_file = os.path.join(results_dir, 'history.pickle')
	with open(history_file, 'wb') as f:
		pickle.dump(ae.train_history_, f, -1)


def main():
	args = parse_command_line()
	model_name = args.model_name
	n_epochs = args.n_epochs
	batch_size = args.batch_size
	checkpoint_every = args.checkpoint_every
	print_every = args.print_every
	rows = args.rows
	columns = args.columns
	learning_rate = args.learning_rate
	dataset = args.dataset
	beta1 = args.beta1
	beta2 = args.beta2

	checkpoint_dir = os.path.join(checkpoint_base_path, model_name,
					dataset, 'caes')
	results_dir = os.path.join(results_base_path, model_name,
					dataset, 'caes')

	recursion_limit = 10000
	print("Setting recursion limit to {rl}".format(rl = recursion_limit))
	sys.setrecursionlimit(recursion_limit)

	X = [batch_size, 1, rows, columns]
	ae = create_caes(X, model_name, learning_rate, beta1, beta2, n_epochs)
	train(ae, X, print_every, checkpoint_every, checkpoint_dir,
			n_epochs, dataset, model_name)

	#weights = models.get_weights(ae)

	dump_results(ae, results_dir)


if __name__ == '__main__':
	main()


"""
def train():
    model = None
    batch_size=128
    if os.path.exists(os.path.join(dataRoot, "model.json")):
        with open(os.path.join(dataRoot, "model.json"), 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
            model.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
    else:
        model = build_model(n_targets)
        model_json_string = model.to_json()
        with open(os.path.join(dataRoot, "model.json"), "w") as f:
            f.write(model_json_string)

    if os.path.exists(os.path.join(dataRoot, "model.h5")):
        print('Loading model from previously saved weights')
        model = load_model(os.path.join(dataRoot, "model.h5"))

    for i in range(30):
        batch_count = 0
        while(1):
            start = batch_count*batch_size
            X_train, y_train = None, None
            try:
                X_train, y_train = get_batch(start, dataType='train',
                                             batch_size=64)
            except Exception as e:
                print('Error occurred while Getting Batch. Skipping this '
                      'batch.')
                traceback.print_stack()
                batch_count += 1
                continue
            if X_train is None or y_train is None:
                break
            [loss, accuracy] = model.train_on_batch(X_train, y_train)
            batch_count += 1
            if batch_count % 1 == 0 and batch_count != 0:
                print('Itr ' + str(batch_count) + '\ttraining loss: ' + str(loss) +
                '\ttraining accuracy: ' + str(accuracy))
            if batch_count % 100 == 0 and batch_count != 0:
                model.save(os.path.join(dataRoot, 'model.h5'))
                print('Saving trained model')
                try:
                    X_val, y_val = get_batch(0, dataType='val', batch_size=100)
                    [loss, accuracy] = model.test_on_batch(X_val, y_val)
                    print('testing loss: ' + str(loss))
                    print('testing accuracy: ' + str(accuracy))
                except Exception as e:
                    print('Error occurred while Getting Batch. Skipping this '
                          'batch.')
                    traceback.print_stack()
                    continue
                # save as JSON


        if i % 2 == 0 and i != 0:
            model.save(os.path.join(dataRoot, 'model.h5'))
            # save as JSON
            print('Saving trained model')
            try:
                X_val, y_val = get_batch(0, dataType='val', batch_size=100)
                [loss, accuracy] = model.test_on_batch(X_val, y_val)
                print('testing loss: ' + str(loss))
                print('testing accuracy: ' + str(accuracy))
            except Exception as e:
                print('Error occurred while Getting Batch. Skipping this '
                      'batch.')
                traceback.print_stack()
                continue
"""
