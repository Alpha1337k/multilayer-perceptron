import math
from typing import List
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

def transfer_derivative(output) -> float:
	return output * (1.0 - output)

def activate(weights, bias, X):
	activation = bias

	for weight, x in zip(weights, X):
		activation += x * weight
	
	return activation

@validate_call(config=model_config)
def train(network: Network, dataset: Dataset):
	weights: List[List[List[float]]] = []
	biases = [-0.5] * len(network.layers)

	prev_layer_size = 1

	for layer in network.layers:
		layer_weights = []

		for i in range(layer.size):
			layer_weights.append(np.array([1 + i * 0.1] * prev_layer_size))
		prev_layer_size = layer.size

		weights.append(layer_weights)

	# print(weights, biases)

	iterations = 1
	l_rate = 0.2

	for i in range(iterations):
		for item_x, item_y in zip(dataset.X, dataset.Y):
			store = InputStore(inputs=item_x, network=network)

			for layer_idx, layer in enumerate(network.layers):
				for node_idx in range(layer.size):
					activation = activate(weights[layer_idx][node_idx], biases[layer_idx], store.get_inputs(layer_idx, node_idx))

					output = 1 / (1 + math.e ** (-activation))

					print(layer_idx, node_idx, store.get_inputs(layer_idx, node_idx), activation, output)

					store.set_inputs(layer_idx, node_idx, output)

			error_store = InputStore(inputs=[], network=network)
			delta_store = InputStore(inputs=[], network=network)

			for layer_idx in reversed(range(len(network.layers))):
				layer = network.layers[layer_idx]
				outputs = store.get_layer(layer_idx)

				if (layer_idx == len(network.layers) - 1):
					for neuron_idx in range(layer.size):
						error_store.set_inputs(layer_idx, neuron_idx, outputs[neuron_idx] - item_y)
				else:
					for neuron_idx in range(layer.size):
						error = 0.0

						for neuron_weights in weights[layer_idx + 1]:
							error += neuron_weights[neuron_idx] * delta_store.get_output(layer_idx, neuron_idx)

						error_store.set_inputs(layer_idx, neuron_idx, error)

				for neuron_idx in range(layer.size):
					error = error_store.get_output(layer_idx, neuron_idx)
					output = store.get_output(layer_idx, neuron_idx)

					delta_store.set_inputs(layer_idx, neuron_idx, error * transfer_derivative(output))

			for i, layer in enumerate(network.layers):
				print(i)

				for n_idx, neuron in enumerate(weights[i]):
					inputs = store.get_inputs(i, n_idx)
					for weight_idx in range(len(neuron)):
						neuron[weight_idx] -= l_rate * delta_store.get_output(i, n_idx) * inputs[weight_idx]
					biases[layer_idx] -= l_rate * delta_store.get_output(i, n_idx)

	for i, layer in enumerate(weights):
		for node in weights:
			print(i, node)



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