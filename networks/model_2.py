from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, \
				UpSampling2D, BatchNormalization
from keras.models import Model, load_model


def get_cnn_parameters():
	return {
		'rows': 256,
		'columns': 256,
		'n_epochs': 30,
		'batch_size': 64,
		'learning_rate': 0.0001,
		'beta1': 0.9,
		'beta2': 0.999,
		'activation': 'relu'
	}

def get_caes_parameters():
	ret = get_cnn_parameters()
	ret['n_epochs'] = 30

	return ret

def get_cnn_network(model_name):
	model, _ = load_model(model_name)
	params = get_caes_parameters()
	pass

def get_caes_network():
	params = get_cnn_parameters()

	input_layer = Input(name = 'input', shape = (256, 256, 1))

	x = Convolution2D(name = 'enc_conv2D_1',
			nb_filter = 96, nb_row = 7, nb_col = 7,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (4, 4))(input_layer)

	x = MaxPooling2D(name = 'enc_maxPool_1',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batcnNorm_1')(x)

	x = Convolution2D(name = 'enc_conv2D_2',
			nb_filter = 256, nb_row = 5, nb_col = 5,
			activation = params['activation'],
			border_mode='same',
			subsample = (1, 1))(x)

	x = MaxPooling2D(name = 'enc_maxPool_2',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batcnNorm_2')(x)

	x = Convolution2D(name = 'enc_conv2D_3',
			nb_filter = 384, nb_row = 3, nb_col = 3,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = Convolution2D(name = 'enc_conv2D_4',
			nb_filter = 256, nb_row = 3, nb_col = 3,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = MaxPooling2D(name = 'enc_maxPool3',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	# This is the middle. This is where we want to "chop" for the CNN
	encoded = BatchNormalization(name = 'enc_batcnNorm_3')(x)

	x = UpSampling2D(name = 'dec_upsample2D_1',
			size = (2, 2))(encoded)

	x = Convolution2D(name = 'dec_conv2D_1',
			nb_filter = 384, nb_row = 3, nb_col = 3,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = Convolution2D(name = 'dec_conv2D_2',
			nb_filter = 256, nb_row = 3, nb_col = 3,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = BatchNormalization(name = 'dec_batchNorm_1')(x)

	x = UpSampling2D(name = 'dec_upsample2D_2',
			size = (2, 2))(x)

	x = Convolution2D(name = 'dec_conv2D_3',
			nb_filter = 96, nb_row = 5, nb_col = 5,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = BatchNormalization(name = 'dec_batchNorm_2')(x)

	x = UpSampling2D(name = 'dec_upsample2D_3',
			size = (2, 2))(x)

	x = Convolution2D(name = 'dec_conv2D_4',
			nb_filter = 1, nb_row = 7, nb_col = 7,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	decoded = UpSampling2D(name = 'dec_upsample2D_4',
			size = (4, 4))(x)

	return input_layer, decoded, params

