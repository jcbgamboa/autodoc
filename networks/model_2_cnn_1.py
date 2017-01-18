import sys
import model_2 as model

def get_cnn_parameters():
	ret = model.get_caes_parameters()
	ret['n_epochs'] = 10

	return ret

def get_cnn_network(caes_name):
	cnn_name = get_cnn_name(caes_name)
	if cnn_name is None:
		print("Invalid network name")
		sys.exit()

	input_layer, net, params = model.get_cnn_network(cnn_name)
	params = get_cnn_network()

	return input_layer, net, params

