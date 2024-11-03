
import math
import pandas as pd
from pydantic import ConfigDict, validate_call
from sklearn.metrics import accuracy_score

from train import forward_propagate
from utils.dataset import Dataset
from utils.weights import read_weights
from utils.network import Network
from utils.layer import Layer

model_config = ConfigDict(arbitrary_types_allowed=True)

@validate_call(config=model_config)
def predict(weights_fd, dataset_fd, network_fd):

	weights_store = read_weights(weights_fd)
	dataset = Dataset.load(pd.read_csv(dataset_fd), weights_store)
	network = eval(network_fd.read())

	y_pred = []
	log_loss = 0
	total_correct = 0

	for item_x, item_y in zip(dataset.X, dataset.Y):
		outputs = forward_propagate(network, weights_store.weights, weights_store.biases, item_x)

		expected = [1 if item_y == 0 else 0, 1 if item_y == 1 else 0]
		res = outputs[-1]

		if (res[0] > res[1]):
			y_pred.append(0)
		else:
			y_pred.append(1)

		for x, (exp, output) in enumerate(zip(expected, outputs[-1])):
			log_loss += -exp * math.log(output + 1e-15) - (1 - exp) * math.log(1 - output + 1e-15)

	print(f"--- RESULTS ---")
	print(f"{100 * accuracy_score(dataset.Y, y_pred):.2f}% correct.")
	print(f"{accuracy_score(dataset.Y, y_pred, normalize=False):.2f} out of {len(dataset.Y)} correct.")
	print(f"Log loss: {log_loss / len(dataset.X)}")
	print(f"--- RESULTS ---")

