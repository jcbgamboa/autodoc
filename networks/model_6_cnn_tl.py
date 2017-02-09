import sys, os
import model_6 as model

# For some reason, it isn't reading its __init__.py file
from . import get_cnn_name, networks_base_path

def get_cnn_parameters():
	ret = model.get_caes_parameters()
	ret['n_epochs'] = 100

	return ret

def get_cnn_network(caes_net_name):
	cnn_name = get_cnn_name(caes_net_name)

	if cnn_name is None:
		print("Invalid network name")
		sys.exit()

	caes_name = os.path.join(networks_base_path,
			cnn_name, 'rvl-cdip/cnn/model.h5')

	input_layer, net, params = model.get_cnn_network(caes_name,
						train_layers = False)
	params = get_cnn_parameters()

	return input_layer, net, params

