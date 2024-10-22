import math
import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from utils.dataset import Dataset
from utils.input_store import InputStore
from utils.layer import Layer
from utils.network import Network

model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=model_config)
def run_sklearn(df: pd.DataFrame):
	df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)

	X = df.iloc[:, 2:]
	Y = df.iloc[:, 1]

	X_Train = X[:-50]
	Y_Train = Y[:-50]

	X_Test = X[-50:]
	Y_Test = Y[-50:]

	clf = MLPClassifier(hidden_layer_sizes=(30, 30),
						random_state=5,
						verbose=True,
						learning_rate_init=0.01)
	
	clf.fit(X_Train, Y_Train)

	ypred = clf.predict(X_Test)

	print(ypred, accuracy_score(Y_Test, ypred))

def activate(weights, bias, X):
	activation = bias

	for weight, x in zip(weights, X):
		activation += x * weight
	
	return activation

@validate_call(config=model_config)
def train(network: Network, dataset: Dataset):
	weights = []
	biases = [-0.5] * len(network.layers)

	prev_layer_size = 1

	for layer in network.layers:
		layer_weights = []

		for i in range(layer.size):
			layer_weights.append(np.array([1] * prev_layer_size))
		prev_layer_size = layer.size

		weights.append(layer_weights)

	# print(weights, biases)

	iterations = 1

	for i in range(iterations):
		for item in dataset.X:

			store = InputStore(inputs=item, network=network)

			for l, layer in enumerate(network.layers):
				for node_idx in range(layer.size):
					activation = activate(weights[l][node_idx], biases[l], store.get_inputs(l, node_idx))

					output = 1 / (1 + math.e ** (-activation))

					print(l, node_idx, store.get_inputs(l, node_idx), activation, output)

					store.set_inputs(l, node_idx, output)
			break


@validate_call(config=model_config)
def run(df: pd.DataFrame):
	dataset = Dataset.make(df, 20)

	# print(len(dataset.X), len(dataset.Y))
	# print(len(dataset.XValidate), len(dataset.YValidate))

	network = Network(layers = [
		# in
		Layer(size=30, activation='sigmoid'),
		# hidden
		Layer(size=30, activation='sigmoid'),
		Layer(size=30, activation='sigmoid'),
		# out
		Layer(size=2, activation='softmax'),
	])

	print(network)

	train(network, dataset)