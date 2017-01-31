import random
import argparse

import math

def get_all_paths_from_dataset(file_name):
	with open(file_name, 'r') as data_file :
		files_list = data_file.readlines()

	ret = {}
	for i, line in enumerate(files_list):
		img_path, lbl = line.strip().split()

		if (lbl in ret):
			ret[lbl].append(img_path)
		else:
			ret[lbl] = [img_path]

	return ret

def select_n_samples_from_dict(data_dict, n):
	ret = {}
	for lbl in data_dict.keys():
		elems_list = data_dict[lbl]
		ret[lbl] = []
		for i in range(n):
			sample = random.choice(elems_list)
			ret[lbl].append(sample)
			elems_list.remove(sample)

	return ret

def partition_dict_proportionally(data_dict, proportion = 0.2):
	# `proportion` is expected to be a number between 0 and 1, and
	# represents how much of the `data_dict` should be transferred to the
	# new dictionary
	#
	# It is not clear how this partition should happen: if it should take
	# proportionally 20%/80% for each one of the lists in `data_dict`, or
	# if it should simply randomly choose any elements of any list (not
	# caring if the sizes of the lists will be proportional).
	#
	# Here, we partition each list proportionally (because it seems more
	# reasonable).
	#
	# TODO: This is VERY similar to `select_n_samples_from_dict()`.

	ret = {}
	for lbl in data_dict.keys():
		elems_list = data_dict[lbl]

		# I need the cast to `int` because `math.ceil()` returns float
		target = int(math.ceil(len(elems_list) * proportion))
		ret[lbl] = []
		for i in range(target):
			sample = random.choice(elems_list)
			ret[lbl].append(sample)
			elems_list.remove(sample)

	return ret

def dump_dict_to_file(data_dict, file_name):
	with open(file_name, 'w') as outfile:
		for lbl in data_dict.keys():
			elems_list = data_dict[lbl]
			for elem in elems_list:
				outfile.write(elem + " " + lbl + "\n")

def main():
	args = parse_command_line()

	data_dict = get_all_paths_from_dataset(args.dataset)
	train = select_n_samples_from_dict(data_dict, args.n_elements)

	# The test set is composed by the elements we didn't select
	dump_dict_to_file(data_dict, args.output_test)

	# The selected elements are further divided into `train` and `validate`
	# in a 80% / 20% proportion (respectively)
	validate = partition_dict_proportionally(train, 0.2)

	dump_dict_to_file(validate, args.output_validate)
	dump_dict_to_file(train, args.output_train)

def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Randomly generates datasets.')

	parser.add_argument('dataset', metavar = 'dataset', type = str,
		help = 'Dataset .txt to be preprocessed.')

	parser.add_argument('n_elements', metavar = 'n_elements', type = int,
		help = 'Specify the number of elements to be sampled from' + \
			'each class.')

	parser.add_argument('output_train', metavar = 'output_train', type = str,
		help = 'Specify the name of the output training file.')

	parser.add_argument('output_validate', metavar = 'output_validate',
		type = str,
		help = 'Specify the name of the output validate file.')

	parser.add_argument('output_test', metavar = 'output_test', type = str,
		help = 'Specify the name of the output test file.')

	return parser.parse_args()

if __name__ == '__main__':
	main()
