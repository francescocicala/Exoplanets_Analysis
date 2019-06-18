from models.simple_convolutional import Simple_Convolutional
import torch


MODELS = {
	'simple_conv': Simple_Convolutional
	}


def init_model(model_name, input_shape, path, device=None, options=None):	
	if options is None:
		options = {}

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = MODELS[model_name](input_shape, **options).to(device)
	torch.save(model, path)