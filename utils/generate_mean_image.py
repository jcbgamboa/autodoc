import argparse
import os

#import skimage.io as skio
import scipy.misc as skio
import numpy as np

import dataloader as dl

# TODO: Maybe I can use the following code (from Wikipedia)
# def online_variance(data):
#    n = 0
#    mean = 0.0
#    M2 = 0.0
#     
#    for x in data:
#        n += 1
#        delta = x - mean
#        mean += delta/n
#        delta2 = x - mean
#        M2 += delta*delta2
#
#    if n < 2:
#        return float('nan')
#    else:
#        return M2 / (n - 1)

def calculate_statistics(ds):
	# Strongly based on:
	# http://stackoverflow.com/questions/10365119/mean-value-and-standard-deviation-of-a-very-huge-data-set

	# I can just hope this won't overflow
	images_sum = np.zeros([ds.resize[0], ds.resize[1], 1],
						dtype = np.float64)
	images_squared_sum = np.zeros([ds.resize[0], ds.resize[1], 1],
						dtype = np.float64)
	for b in ds:
		curr_batch_sum = b[0].sum(axis=0)
		images_sum += curr_batch_sum
		images_squared_sum += curr_batch_sum ** 2
		count = b[0].shape

	print("Count:", count)

	images_mean = images_sum / len(ds.data_list)
	images_var  = np.sqrt((images_squared_sum / len(ds.data_list) -
						(images_mean ** 2)))
	return images_mean, images_var


def dump_statistics(output_path, mean, variance):
	mean_file = os.path.join(output_path, 'mean_image.tiff')
	variance_file = os.path.join(output_path, 'variance_image.tiff')

	mean_image = skio.toimage(mean.squeeze(),
				cmin = 0.0, cmax=256).save(mean_file)
	variance_image = skio.toimage(mean.squeeze(),
				cmin = 0.0, cmax=256).save(mean_file)


def main():
	args = parse_command_line()

	ds = dl.Dataset(args.dataset)
	ds.resize = [args.rows, args.columns]
	ds.batch_size = 256
	ds.mode = 'train'

	mean, variance = calculate_statistics(ds)

	dump_statistics(args.output_path, mean, variance)


def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Generates the mean image of the given dataset.')

	parser.add_argument('dataset', metavar = 'dataset',
		type = str, help = 'Name of the dataset to be processed.')

	parser.add_argument('output_path', metavar = 'output_path',
		type = str, help = 'Name of the dataset to be processed.')

	parser.add_argument('rows', metavar = 'rows', type = int,
		help = 'Number of rows in the image of the resized image.')

	parser.add_argument('columns', metavar = 'columns', type = int,
		help = 'Number of columns in the image of the resized image.')

	return parser.parse_args()

if __name__ == '__main__':
	main()
