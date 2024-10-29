import json
from pandas import Series
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    field_validator,
    model_validator,
    validate_call,
)
from utils.dataset import Dataset
from typing import List, Optional, Any, Tuple
import numpy as np
import io

from utils.layer import Layer
from utils.network import Network

class WeightsFile(BaseModel):
	weights: List[List[List[float]]]
	mean: List[float]
	std: List[float]
	biases: List[float]
	network: Network

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def save_weights(file, 
				 weights: List[List[List[float]]], 
				 biases: List[float], 
				 mean: np.ndarray, 
				 std: np.ndarray,
				 network: Network
				 ):
	
	data = {
		"weights": weights,
		"biases": biases,
		"mean": mean.tolist(),
		"std": std.tolist(),
		"layers": [{"size": layer.size, "activation": layer.activation} for layer in network.layers] 
	}

	file.write(json.dumps(data, indent=4))


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def read_weights(file: io.TextIOWrapper) -> WeightsFile:
	parsed = json.loads(file.read())

	network = Network(layers=[
		Layer(size=item['size'], activation=item['activation']) for item in parsed["layers"]
	])

	return WeightsFile(
		weights=parsed["weights"],
		std=parsed["std"],
		mean=parsed["mean"],
		biases=parsed["biases"],
		network=network
	)