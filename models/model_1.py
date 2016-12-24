# Import all stuff from Lasagne
# TODO: somehow clean this
from lasagne.layers import InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.nonlinearities import sigmoid, rectify #, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')


def get_layers(X, n_channels):
	conv_num_filters1 = 9
	conv_num_filters2 = 16
	filter_size1 = 7
	filter_size2 = 6
	pool_size = 2
	dense_mid_size = 128
	pad_in = 'valid'
	pad_out = 'full'

	middle_size_x = (((X.shape[2] - (filter_size1 - 1))/pool_size) -
				(filter_size2 - 1))/pool_size
	middle_size_y = (((X.shape[3] - (filter_size1 - 1))/pool_size) -
				(filter_size2 - 1))/pool_size

	encode_size = conv_num_filters2 * middle_size_x * middle_size_y - 16


	# Notice that, by default, Lasagne already uses a Xavier initialization
	return [
	    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters1,
				'nonlinearity': sigmoid,
				'filter_size': filter_size1, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'pool_size': pool_size}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters2,
				'nonlinearity': sigmoid,
				'filter_size': filter_size2, 'pad': pad_in}),
	    (MaxPool2DLayerFast, {'pool_size': pool_size}),
	    (ReshapeLayer, {'shape': (([0], -1))}),
	    (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
	    (DenseLayer, {'num_units': conv_num_filters2 * middle_size_x * middle_size_y}),
	    (ReshapeLayer, {'shape': (([0], conv_num_filters2,
				middle_size_x, middle_size_y))}),
	    (Upscale2DLayer, {'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'num_filters': conv_num_filters1,
				'nonlinearity': sigmoid,
				'filter_size': filter_size2, 'pad': pad_out}),
	    (Upscale2DLayer, {'scale_factor': pool_size}),
	    (Conv2DLayerFast, {'num_filters': X.shape[1],
				'nonlinearity': sigmoid,
				'filter_size': filter_size1, 'pad': pad_out}),
	    (ReshapeLayer, {'shape': (([0], -1))}),
	]

