import sys

import numpy as np

import keras
from keras.models import load_model

from PIL import Image


def load_images(image_file_paths):
	img_list = []

	for i in image_file_paths:
		print("Will open image: ", i)
		img = Image.open(i)
		img = img.resize((256, 256), Image.BICUBIC)

		img_bw = img.convert('L')
		img_bw = np.asarray(img_bw, dtype = np.uint8)
		img_bw = img_bw[:, :, np.newaxis]
		img_list.append(img_bw)

	img_batch = np.array(img_list)
	return img_batch

def reconstruct(model_file_path, img_batch):
	print("img_batch.shape: ", img_batch.shape)
	print("img_batch[0,:,:,:].shape: ", img_batch[0,:,:,:].shape)

	img = Image.fromarray(img_batch[0,:,:,:].squeeze(), 'L')
	img.show()

	model = load_model(model_file_path)
	result = model.predict(img_batch)

	print("result.shape: ", result.shape)
	print("result[0,:,:,:].shape: ", result[0,:,:,:].shape)

	img = Image.fromarray(result[0,:,:,:].squeeze(), 'L')
	img.show()

if __name__ == '__main__':
	print("Args", sys.argv)

	img_batch = load_images([sys.argv[2]])
	reconstruct(sys.argv[1], img_batch)

