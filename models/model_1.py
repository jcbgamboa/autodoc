# Import all stuff from Lasagne
# TODO: somehow clean this
from lasagne.layers import InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import sigmoid, rectify #, leaky_rectify, tanh
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')


def get_layers(X):
	conv_num_filters1 = 9
	conv_num_filters2 = 16
	filter_size1 = 7
	filter_size2 = 6
	pool_size = 2
	dense_mid_size = 128
	pad_in = 'valid'
	pad_out = 'full'

	middle_size_x = (((X[2] - (filter_size1 - 1))/pool_size) -
				(filter_size2 - 1))/pool_size
	middle_size_y = (((X[3] - (filter_size1 - 1))/pool_size) -
				(filter_size2 - 1))/pool_size

	encode_size = 500
	non_linearity = rectify


	# Notice that, by default, Lasagne already uses a Xavier initialization
	return [
	    (InputLayer, {'name': 'input',
				'shape': (None, X[1], X[2], X[3])}),
	    (Conv2DLayerFast, {'name': 'e_conv2D_1',
				'num_filters': conv_num_filters1,
				'nonlinearity': rectify,
				'filter_size': filter_size1, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'name': 'maxpool2D_1',
				'pool_size': pool_size}),
	    (Conv2DLayerFast, {'name': 'e_conv2D_2',
				'num_filters': conv_num_filters2,
				'nonlinearity': rectify,
				'filter_size': filter_size2, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'name': 'maxpool2D_2',
				'pool_size': pool_size}),
	    (ReshapeLayer, {'name': 'reshape_1',
				'shape': (([0], -1))}),
	    (DenseLayer, {'name': 'mid_dense',
				'name': 'encode', 'num_units': encode_size}),
	    (DenseLayer, {'name': 'd_dense',
				'num_units': conv_num_filters2 *
					middle_size_x * middle_size_y}),
	    (ReshapeLayer, {'name': 'reshape_2',
				'shape': (([0], conv_num_filters2,
					middle_size_x, middle_size_y))}),
	    (Upscale2DLayer, {'name': 'upscale2D_1',
				'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'name': 'd_conv2D_1',
				'num_filters': conv_num_filters1,
				'nonlinearity': rectify,
				'filter_size': filter_size2, 'pad': pad_out}),
	    (Upscale2DLayer, {'name': 'upscale2D_2',
				'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'name': 'd_conv2D_2',
				'num_filters': X[1],
				'nonlinearity': rectify,
				'filter_size': filter_size1, 'pad': pad_out}),
	    (ReshapeLayer, {'name': 'reshape_3',
				'shape': (([0], -1))}),
	]

