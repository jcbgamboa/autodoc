def get_weights(ae):
	ret = []
	for l in ae.layers_.values():
		if (l.name.startswith('e_')):
			ret.append((l.W.get_value(), l.b.get_value()))

	return ret

def get_input_shape(ae):
	return ae.layers_['input'].shape
