from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, \
			UpSampling2D, BatchNormalization, Flatten, Dropout
from keras.models import Model, load_model

def get_caes_parameters():
	return {
		'rows': 256,
		'columns': 256,
		'n_epochs': 25,
		'batch_size': 64,
		'learning_rate': 0.0001,
		'beta1': 0.9,
		'beta2': 0.999,
		'activation': 'relu'
	}

def get_cnn_network(network_name, with_dropout=None): #, train_layers=True):
	print("inside get_cnn_network", network_name)
	model = load_model(network_name)

	# It is ok to get the `caes` params, because the network will need to
	# be created with the same structure anyway
	params = get_caes_parameters()

	# Index `[1:]` takes all elements except the first one (which is "None")
	input_shape = model.layers[0].input_shape[1:]

	input_layer = Input(name = 'input', shape = input_shape)

	x = Convolution2D(name = 'enc_conv2D_1',
		#trainable = train_layers,
		weights = model.get_layer('enc_conv2D_1').get_weights(),
		nb_filter = 96, nb_row = 7, nb_col = 7,
		activation = params['activation'],
		border_mode = 'same',
		subsample = (4, 4))(input_layer)

	x = BatchNormalization(name = 'enc_batchNorm_1',
		#trainable = train_layers,
		weights = model.get_layer('enc_batchNorm_1').get_weights())(x)

	x = MaxPooling2D(name = 'enc_maxPool_1',
		pool_size = (2, 2),
		border_mode = 'same')(x)

	x = Convolution2D(name = 'enc_conv2D_2',
		#trainable = train_layers,
		weights = model.get_layer('enc_conv2D_2').get_weights(),
		nb_filter = 256, nb_row = 5, nb_col = 5,
		activation = params['activation'],
		border_mode='same',
		subsample = (1, 1))(x)

	x = BatchNormalization(name = 'enc_batchNorm_2',
		#trainable = train_layers,
		weights = model.get_layer('enc_batchNorm_2').get_weights())(x)

	x = MaxPooling2D(name = 'enc_maxPool_2',
		pool_size = (2, 2),
		border_mode = 'same')(x)

	x = Convolution2D(name = 'enc_conv2D_3',
		#trainable = train_layers,
		weights = model.get_layer('enc_conv2D_3').get_weights(),
		nb_filter = 384, nb_row = 3, nb_col = 3,
		activation = params['activation'],
		border_mode = 'same',
		subsample = (1, 1))(x)

	x = BatchNormalization(name = 'enc_batchNorm_3',
		#trainable = train_layers,
		weights = model.get_layer('enc_batchNorm_3').get_weights())(x)

	x = Convolution2D(name = 'enc_conv2D_4',
		#trainable = train_layers,
		weights = model.get_layer('enc_conv2D_4').get_weights(),
		nb_filter = 256, nb_row = 3, nb_col = 3,
		activation = params['activation'],
		border_mode = 'same',
		subsample = (1, 1))(x)

	x = BatchNormalization(name = 'enc_batchNorm_4',
		#trainable = train_layers,
		weights = model.get_layer('enc_batchNorm_4').get_weights())(x)

	x = MaxPooling2D(name = 'enc_maxPool3',
		pool_size = (2, 2),
		border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batchNorm_5',
		#trainable = train_layers,
		weights = model.get_layer('enc_batchNorm_5').get_weights())(x)

	# These are CNN specific
	net = Flatten()(x)
	net = Dense(1024, activation = params['activation'])(net)
	if (with_dropout):
		net = Dropout(0.5)(net)
	net = BatchNormalization()(net)
	net = Dense(512, activation = params['activation'])(net)
	if (with_dropout):
		net = Dropout(0.5)(net)
	net = BatchNormalization()(net)

	# I need to specify what to return here
	return input_layer, net, params


def get_caes_network(with_dropout = None):
	params = get_caes_parameters()

	input_layer = Input(name = 'input', shape = (256, 256, 1))

	x = Convolution2D(name = 'enc_conv2D_1',
		nb_filter = 96, nb_row = 7, nb_col = 7,
		activation = params['activation'],
		border_mode = 'same',
		subsample = (4, 4))(input_layer)

	x = BatchNormalization(name = 'enc_batchNorm_1')(x)

	x = MaxPooling2D(name = 'enc_maxPool_1',
		pool_size = (2, 2),
		border_mode = 'same')(x)

	x = Convolution2D(name = 'enc_conv2D_2',
		nb_filter = 256, nb_row = 5, nb_col = 5,
		activation = params['activation'],
		border_mode='same',
		subsample = (1, 1))(x)

	x = BatchNormalization(name = 'enc_batchNorm_2')(x)

	x = MaxPooling2D(name = 'enc_maxPool_2',
		pool_size = (2, 2),
		border_mode = 'same')(x)

	x = Convolution2D(name = 'enc_conv2D_3',
		nb_filter = 384, nb_row = 3, nb_col = 3,
		activation = params['activation'],
		border_mode = 'same',
		subsample = (1, 1))(x)

	x = BatchNormalization(name = 'enc_batchNorm_3')(x)

	x = Convolution2D(name = 'enc_conv2D_4',
		nb_filter = 256, nb_row = 3, nb_col = 3,
		activation = params['activation'],
		border_mode = 'same',
		subsample = (1, 1))(x)

	x = BatchNormalization(name = 'enc_batchNorm_4')(x)

	x = MaxPooling2D(name = 'enc_maxPool3',
		pool_size = (2, 2),
		border_mode = 'same')(x)

	x = BatchNormalization(name = 'enc_batchNorm_5')(x)

	# These are CNN specific
	net = Flatten()(x)
	net = Dense(1024, activation = params['activation'])(net)
	if (with_dropout):
		net = Dropout(0.5)(net)
	net = BatchNormalization()(net)
	net = Dense(512, activation = params['activation'])(net)
	if (with_dropout):
		net = Dropout(0.5)(net)
	net = BatchNormalization()(net)

	return input_layer, net, params

