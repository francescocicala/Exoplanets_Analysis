import numpy as np
import torch
from torch.utils.data import DataLoader

from exoplanets_analysis.process_data import ExoplanetDataset
from exoplanets_analysis.initialize_model import init_model
from exoplanets_analysis.train import train_step
print("modules imported!\n")


if __name__ == '__main__':
	moving_average = 50
	batch_size = 32
	model_path = "tests/conv1.pt"
	model_type = 'simple_conv'

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device: {}.".format(device))


	### GENERATE TRAIN DATASET ###
	train_dataset = torch.load("data/train_dataset.pt")

	kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
	trainloader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
	print("trainloader created!\n")
	input_shape = trainloader.dataset.X.shape[2]
	print("Input shape: {}.".format(input_shape))


	### GENERATE MODEL ###
	init_model(model_type, input_shape, path=model_path, device=device)
	print("Model of type '{}' have been initialized!\n".format(model_type))


	### TRAIN MODEL ###
	model = torch.load(model_path).to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	epochs = 20
	for epoch in range(epochs):
		train_step(model, trainloader, criterion, optimizer, epoch, device)

	torch.save(model, model_path)
	print("Training completed! Model saved.")


