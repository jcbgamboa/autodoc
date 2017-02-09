import ntpath 
import os


caes_cnn_map = {
	'model_2_cnn': 'model_2',

	'model_5_cnn': 'model_5',
	'model_5_cnn_dropout': 'model_5',
	'model_5_cnn_dropout_nomean': 'model_5',
	'model_5_cnn_nomean': 'model_5',

	'model_6_cnn': 'model_6',
	'model_6_cnn_dropout': 'model_6'
}

networks_base_path = 'data/results/'

def get_cnn_name(caes_name):
	#network_name = ntpath.basename(caes_name)
	#path_name = '/'.join(existGDBPath.split('/')[0:-1])
	#network_file_path = os.path.join(path_name, network_name)
	#return caes_cnn_map[network_file_path]
	return caes_cnn_map[caes_name]

