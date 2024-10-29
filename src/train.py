import math
from os import error
from typing import Callable, List, TypeVar
import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from utils.dataset import Dataset
from utils.input_store import InputStore
from utils.layer import Layer
from utils.network import Network
from random import seed
from random import random

model_config = ConfigDict(arbitrary_types_allowed=True)

T = TypeVar('T')

@validate_call
def make_rand_array(size: int):
	arr = []

	for i in range(size):
		# arr.append(1)
		arr.append((random() - 0.5) * 2)

	return arr

@validate_call
def make_structure(network: Network, init: Callable[[int, int], T]) -> List[List[T]]:
	structure = []
	
	for l_idx, layer in enumerate(network.layers):
		items = [init(l_idx, i) for i in range(layer.size)]
		structure.append(items)

	return structure

@validate_call(config=model_config)
def run_sklearn(df: pd.DataFrame):
	df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)

	X = df.iloc[:, 2:]
	Y = df.iloc[:, 1]

	X_Train = X[:-50]
	Y_Train = Y[:-50]

	X_Test = X[-50:]
	Y_Test = Y[-50:]

	clf = MLPClassifier(hidden_layer_sizes=(8, 8),
						random_state=5,
						verbose=True,
						learning_rate_init=0.01)
	
	clf.fit(X_Train, Y_Train)

	ypred = clf.predict(X_Test)

	print(ypred, accuracy_score(Y_Test, ypred))

def transfer_derivative(output) -> float:
	return output * (1.0 - output)

@validate_call
def activate(weights: List[float], bias: float, X: List[float]) -> float:
	activation = bias

	for weight, x in zip(weights, X):
		activation += x * weight
	
	return activation

@validate_call
def forward_propagate(network: Network, weights: List[List[List[float]]], biases: List[float], X: List[float]):
	inputs = make_structure(network, lambda l, i: 0.0)

	for layer_idx, layer in enumerate(network.layers):
		sm_activation = []
		for node_idx in range(layer.size):
			if (layer_idx == 0):
				input = [X[node_idx]]
			else:
				input = inputs[layer_idx - 1]

			activation = activate(weights[layer_idx][node_idx], biases[layer_idx], input)

			if (layer.activation == 'sigmoid'):
				output = 1 / (1 + math.e ** (-activation))
				inputs[layer_idx][node_idx] = output
			elif layer.activation == 'softmax':
				sm_activation.append(math.e ** activation)
			else: raise Exception("InvalidActivator")

			# print(layer_idx, node_idx, store.get_inputs(layer_idx, node_idx), activation, output)

		if (layer.activation == 'softmax'):
			total_activation = sum(sm_activation)

			inputs[layer_idx] = [activation / (1 + total_activation) for activation in sm_activation]


	return inputs

@validate_call
def backward_propagate(network: Network, weights: List[List[List[float]]], outputs: List[List[float]], expected: List[int]):
	deltas = make_structure(network, lambda l, i: 0.0)
	errors = make_structure(network, lambda l, i: 0.0)

	for layer_idx in reversed(range(len(network.layers))):
		layer = network.layers[layer_idx]
		layer_output = outputs[layer_idx] 

		if (layer_idx == len(network.layers) - 1):
			for neuron_idx in range(layer.size):
				errors[layer_idx][neuron_idx] = layer_output[neuron_idx] - expected[neuron_idx]
		else:
			for neuron_idx in range(layer.size):
				error = 0.0

				for forward_node_idx, forward_weight in enumerate(weights[layer_idx + 1]):
					error += forward_weight[neuron_idx] * deltas[layer_idx + 1][forward_node_idx]

				errors[layer_idx][neuron_idx] = error

		for neuron_idx in range(layer.size):
			error = errors[layer_idx][neuron_idx]
			output = outputs[layer_idx][neuron_idx]

			deltas[layer_idx][neuron_idx] = error * transfer_derivative(output)
	
	return deltas, errors

@validate_call
def update_weights(
	network: Network, 
	weights: List[List[List[float]]], 
	biases: List[float], 
	deltas: List[List[float]], 
	outputs: List[List[float]],
	X: List[float]
):
	l_rate = 0.75

	for layer_idx, layer in enumerate(network.layers):
		for n_idx, neuron in enumerate(weights[layer_idx]):
			inputs: List[float] = []

			if (layer_idx == 0):
				inputs = [X[n_idx]]
			else:
				inputs = outputs[layer_idx - 1]

			for input_idx in range(len(inputs)):
				weights[layer_idx][n_idx][input_idx] -= l_rate * deltas[layer_idx][n_idx] * inputs[input_idx]

			# biases[layer_idx] -= l_rate * deltas[layer_idx][n_idx]

	return weights, biases


@validate_call(config=model_config)
def train(network: Network, dataset: Dataset):
	weights = make_structure(network, lambda l, i: make_rand_array(network.layers[l - 1].size) if l > 0 else [random()])
	biases = [0.] * len(network.layers)

	iterations = 200

	for i in range(iterations):
		sum_error = 0.0

		for item_x, item_y in zip(dataset.X, dataset.Y):
			outputs = forward_propagate(network, weights, biases, item_x)

			# for (output, weight) in zip(outputs, weights):
			# 	print(output, "\n||\n", weight, "\n")

			# exit(1)

			expected = [1 if item_y == 0 else 0, 1 if item_y == 1 else 0]

			sum_error += sum((exp - output) ** 2 for exp, output in zip(expected, outputs[-1]))

			deltas, errors = backward_propagate(network, weights, outputs, expected)

			# print(item_y, outputs[-1], deltas[-1], errors[-1])

			weights, biases = update_weights(network, weights, biases, deltas, outputs, item_x)

		print(f"Epoch {i}, sum_error: {sum_error}, bias: {biases}")

	return weights, biases


@validate_call
def validate(network: Network, weights: List[List[List[float]]], biases: List[float], dataset: Dataset):
	y_pred = []
	for item_x, item_y in zip(dataset.XValidate, dataset.YValidate):
		output = forward_propagate(network, weights, biases, item_x)

		res = output[-1]

		if (res[0] > res[1]):
			y_pred.append(0)
		else:
			y_pred.append(1)

	print(f"--- RESULTS ---")
	print(f"{100 * accuracy_score(dataset.YValidate, y_pred):.2f}% correct.")
	print(f"{accuracy_score(dataset.YValidate, y_pred, normalize=False):.2f} out of {len(dataset.YValidate)} correct.")
	print(f"--- RESULTS ---")

	y_pred = []
	for item_x, item_y in zip(dataset.X, dataset.Y):
		output = forward_propagate(network, weights, biases, item_x)

		res = output[-1]

		if (res[0] > res[1]):
			y_pred.append(0)
		else:
			y_pred.append(1)

	print(f"--- RESULTS ---")
	print(f"{100 * accuracy_score(dataset.Y, y_pred):.2f}% correct.")
	print(f"{accuracy_score(dataset.Y, y_pred, normalize=False):.2f} out of {len(dataset.Y)} correct.")
	print(f"--- RESULTS ---")
	


@validate_call(config=model_config)
def run(df: pd.DataFrame):
	dataset = Dataset.make(df, 20)

	# print(len(dataset.X), len(dataset.Y))
	# print(len(dataset.XValidate), len(dataset.YValidate))

	network = Network(layers = [
		# in
		Layer(size=30, activation='sigmoid'),
		# hidden
		Layer(size=20, activation='sigmoid'),
		Layer(size=20, activation='sigmoid'),
		# out
		Layer(size=2, activation='softmax'),
	])

	print(network)

	weights, biases = train(network, dataset)

	validate(network, weights, biases, dataset)

	# run_sklearn(df)