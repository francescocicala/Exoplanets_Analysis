import numpy as np
import torch
from process_data import ExoplanetDataset


if __name__ == '__main__':
	dataset_path = "data/validation_dataset.pt"
	dataset = torch.load(dataset_path)
	X = dataset.X.numpy()
	y = dataset.y

	positives_num = y.sum()
	print("{} has length {}.\n{} are exoplanets.".format(dataset_path, len(dataset), positives_num))
