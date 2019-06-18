import numpy as np
import torch
from torch.utils.data import Dataset

class ExoplanetDataset(Dataset):
    """Exoplanet dataset."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        sample = (int(self.y[idx]), x)
        return sample


def remove_nan(data):
	return data[~np.isnan(data).any(axis=1)]


def standardize(X, epsilon=0.01):
    return (X - np.mean(X, axis=1).reshape(-1, 1)) / (np.std(X, axis=1).reshape(-1, 1) + epsilon)


def totensor(X, unsqueeze=True, dtype=torch.float):
    X_tens = torch.tensor(X, dtype=dtype)
    if unsqueeze:
        X_tens = X_tens.unsqueeze(1)
    return X_tens


def split_X_y(data):
    X = data[:, 0:-1]
    y = data[:, -1]- 1
    return X, y


def apply_moving_average(X, moving_average):
    X_new = []
    for x in X:
        new_x = np.convolve(x, np.ones((moving_average,))/moving_average, mode='valid')
        X_new.append(new_x)

    return np.asarray(X_new)


def generate_dataset(data_path, save_path, moving_average):
    data = np.genfromtxt(data_path, delimiter=',')
    data = remove_nan(data)
    X, y = split_X_y(data)
    X = apply_moving_average(X, moving_average)
    X = standardize(X)
    X = totensor(X)

    dataset = ExoplanetDataset(X, y)
    torch.save(dataset, save_path)



if __name__ == '__main__':
    folder = "../data/"
    moving_average = 50
    generate_dataset(folder+"TRAIN.csv", folder+"train_dataset.pt", moving_average)
    print("train_dataset.pt saved in " + folder + " folder.")

    generate_dataset(folder+"VALIDATION.csv", folder+"validation_dataset.pt", moving_average)
    print("validation_dataset.pt saved in " + folder + " folder.")







    