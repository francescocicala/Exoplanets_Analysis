import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import fbeta_score
from models.simple_convolutional import Simple_Convolutional
from process_data import ExoplanetDataset

def get_probabilities(model, X):
    return F.softmax(model(X), dim=1).detach().numpy()[:, 0]


def classify(probs, threshold):
    return (probs > threshold).astype(int)


def best_threshold_for_fbeta(probs, targets, beta):
	thresholds = np.linspace(0.1, 0.99, 1000)
	fbeta_scores = []
	for t in thresholds:
		preds = classify(probs, t)
		fbeta_scores.append(fbeta_score(targets, preds, beta))
	fbeta_scores = np.asarray(fbeta_scores)
	best_index = np.argmax(fbeta_scores)
	return thresholds[best_index]

if __name__ == '__main__':
	data_path = "data/train_dataset.pt"
	model_path = "tests/conv1.pt"
	beta = 3

	### GENERATE VALIDATION DATASET
	validation_dataset = torch.load(data_path)
	X_val = validation_dataset.X
	targets = validation_dataset.y

	model = torch.load(model_path)

	probs = get_probabilities(model, X_val)

	threshold = best_threshold_for_fbeta(probs, targets, beta)
	preds = classify(probs, threshold)
	fbeta = fbeta_score(targets, preds, beta)
	
	print("The threshold which maximize the f-{} score is {}.\nThe f-{} score is {}.".format(
		beta, threshold, beta, fbeta))
	

