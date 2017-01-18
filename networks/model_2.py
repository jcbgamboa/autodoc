from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, \
				UpSampling2D, BatchNormalization
from keras.models import Model, load_model


def get_caes_parameters():
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

def get_cnn_network(network_name):
	model, _ = load_model(network_name)

	# It is ok to get the `caes` params, because the network will need to
	# be created with the same structure anyway
	params = get_caes_parameters()

	# Index `[1:]` takes all elements except the first one (which is "None")
	input_shape = model.layers[0].input_shape[1:]

	input_layer = Input(name = 'input', shape = input_shape)

	x = Convolution2D(name = 'enc_conv2D_1',
			weights = model.get_layer('enc_conv2D_1'),
			nb_filter = 96, nb_row = 7, nb_col = 7,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (4, 4))(input_layer)

	x = MaxPooling2D(name = 'enc_maxPool_1',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batchNorm_1',
			weights = model.get_layer('enc_batch_1'))(x)

	x = Convolution2D(name = 'enc_conv2D_2',
			weights = model.get_layer('enc_conv2D_2'),
			nb_filter = 256, nb_row = 5, nb_col = 5,
			activation = params['activation'],
			border_mode='same',
			subsample = (1, 1))(x)

	x = MaxPooling2D(name = 'enc_maxPool_2',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batchNorm_2',
			weights = model.get_layer('enc_batch_2'))(x)

	x = Convolution2D(name = 'enc_conv2D_3',
			weights = model.get_layer('enc_conv2D_3'),
			nb_filter = 384, nb_row = 3, nb_col = 3,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = Convolution2D(name = 'enc_conv2D_4',
			weights = model.get_layer('enc_conv2D_4'),
			nb_filter = 256, nb_row = 3, nb_col = 3,
			activation = params['activation'],
			border_mode = 'same',
			subsample = (1, 1))(x)

	x = MaxPooling2D(name = 'enc_maxPool3',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	# This is the middle. This is where we want to "chop" for the CNN
	x = BatchNormalization(name = 'enc_batchNorm_3',
			weights = model.get_layer('enc_batch_3'))(x)

	# These are CNN specific
	net = Flatten()(x)
	net = Dense(512, activation = activation)(net)
	net = Dropout(0.5)(net)
	net = BatchNormalization()(net)
	net = Dense(1024, activation=activation)(net)
	net = Dropout(0.5)(net)
	net = BatchNormalization()(net)

	# I need to specify what to return here
	return input_layer, net, params


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

	x = BatchNormalization(name = 'enc_batchNorm_1')(x)

	x = Convolution2D(name = 'enc_conv2D_2',
			nb_filter = 256, nb_row = 5, nb_col = 5,
			activation = params['activation'],
			border_mode='same',
			subsample = (1, 1))(x)

	x = MaxPooling2D(name = 'enc_maxPool_2',
			pool_size = (2, 2),
			border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batchNorm_2')(x)

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
	encoded = BatchNormalization(name = 'enc_batchNorm_3')(x)

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

