import math
from os import error
from typing import Callable, List, TypeVar
import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from plot import plot_accuracy, plot_log_loss
from utils.dataset import Dataset
from utils.layer import Layer
from utils.network import Network
from random import seed
from random import random

from utils.weights import save_weights

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
	X: List[float],
	learning_rate: float
):
	for layer_idx, layer in enumerate(network.layers):
		for n_idx, neuron in enumerate(weights[layer_idx]):
			inputs: List[float] = []

			if (layer_idx == 0):
				inputs = [X[n_idx]]
			else:
				inputs = outputs[layer_idx - 1]

			for input_idx in range(len(inputs)):
				weights[layer_idx][n_idx][input_idx] -= learning_rate * deltas[layer_idx][n_idx] * inputs[input_idx]

			biases[layer_idx] -= learning_rate * deltas[layer_idx][n_idx]

	return weights, biases

@validate_call(config=model_config)
def	get_validation_scores(network: Network, dataset: Dataset, weights: List[List[List[float]]], biases: List[float]):
	log_loss_score = 0
	accuracy_score = 0

	for item_x, item_y in zip(dataset.XValidate, dataset.YValidate):
		outputs = forward_propagate(network, weights, biases, item_x)


		expected = [1 if item_y == 0 else 0, 1 if item_y == 1 else 0]

		if (outputs[-1][0] > outputs[-1][1] and item_y == 0) or (outputs[-1][1] > outputs[-1][0] and item_y == 1):
			accuracy_score += 1

		for x, (exp, output) in enumerate(zip(expected, outputs[-1])):
			log_loss_score += -exp * math.log(output + 1e-15) - (1 - exp) * math.log(1 - output + 1e-15)

	return log_loss_score / len(dataset.XValidate), accuracy_score / len(dataset.XValidate)



@validate_call(config=model_config)
def train(network: Network, dataset: Dataset, iterations: int, learning_rate: float):
	weights = make_structure(network, lambda l, i: make_rand_array(network.layers[l - 1].size) if l > 0 else [random()])
	biases = [0.] * len(network.layers)

	log_loss_store_train = []
	accuracy_store_train = []
	log_loss_store_validate = []
	accuracy_store_validate = []


	for i in range(iterations):
		log_loss = 0.0
		total_correct = 0

		for item_x, item_y in zip(dataset.X, dataset.Y):
			outputs = forward_propagate(network, weights, biases, item_x)

			# for (output, weight) in zip(outputs, weights):
			# 	print(output, "\n||\n", weight, "\n")

			# exit(1)

			expected = [1 if item_y == 0 else 0, 1 if item_y == 1 else 0]

			if (outputs[-1][0] > outputs[-1][1] and item_y == 0) or (outputs[-1][1] > outputs[-1][0] and item_y == 1):
				total_correct += 1

			# log_loss += sum((exp - output) ** 2 for exp, output in zip(expected, outputs[-1]))

			for x, (exp, output) in enumerate(zip(expected, outputs[-1])):
				log_loss += -exp * math.log(output + 1e-15) - (1 - exp) * math.log(1 - output + 1e-15)


			deltas, errors = backward_propagate(network, weights, outputs, expected)

			# print(item_y, outputs[-1], deltas[-1], errors[-1])

			weights, biases = update_weights(network, weights, biases, deltas, outputs, item_x, learning_rate)

		validation_log_loss, validation_accuracy = get_validation_scores(network, dataset, weights, biases)


		print(f"Epoch {i}, Train: [loss: {log_loss / len(dataset.X)}, accuracy: {total_correct / len(dataset.X)}], validate: [loss: {validation_log_loss}, accuracy: {validation_accuracy}]")

		log_loss_store_train.append(log_loss / len(dataset.X))
		accuracy_store_train.append(total_correct / len(dataset.X))
		log_loss_store_validate.append(validation_log_loss)
		accuracy_store_validate.append(validation_accuracy)

	return weights, biases, log_loss_store_train, accuracy_store_train, log_loss_store_validate, accuracy_store_validate,
								


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
def run(input, output, config_fd, iterations: int, learning_rate: float):
	df = pd.read_csv(input)

	labels = df['diagnosis'].copy()

	df.drop(labels=[ 'diagnosis' ], axis=1)

	mean, std = df.iloc[:,2:].mean().to_numpy(), df.iloc[:,2:].std().to_numpy()

	dataset = Dataset.make(df, 20)

	network = eval(config_fd.read())

	weights, biases, loss_train, accuracy_train, loss_validate, accuracy_validate = train(network, dataset, iterations, learning_rate)

	validate(network, weights, biases, dataset)

	plot_accuracy(accuracy_train, accuracy_validate)
	plot_log_loss(loss_train, loss_validate)

	save_weights(output, weights, biases, mean, std, network)

	# run_sklearn(df)