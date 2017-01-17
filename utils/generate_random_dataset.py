import random
import argparse

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

def dump_dict_to_file(data_dict, file_name):
	with open(file_name, 'w') as outfile:
		for lbl in data_dict.keys():
			elems_list = data_dict[lbl]
			for elem in elems_list:
				outfile.write(elem + " " + lbl + "\n")

def main():
	args = parse_command_line()

	data_dict = get_all_paths_from_dataset(args.dataset)
	samples = select_n_samples_from_dict(data_dict, args.n_elements)
	dump_dict_to_file(samples, args.output)

def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Randomly generates datasets.')

	parser.add_argument('dataset', metavar = 'dataset', type = str,
		help = 'Dataset .txt to be preprocessed.')

	parser.add_argument('n_elements', metavar = 'n_elements', type = int,
		help = 'Specify the number of elements to be sampled from' + \
			'each class.')

	parser.add_argument('output', metavar = 'output', type = str,
		help = 'Specify the name of the output file.')

	return parser.parse_args()

if __name__ == '__main__':
	main()
