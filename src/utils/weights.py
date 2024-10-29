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

class WeightsFile(BaseModel):
	weights: List[List[List[float]]]
	mean: List[float]
	std: List[float]
	biases: List[float]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def save_weights(file, weights: List[List[List[float]]], biases: List[float], mean: np.ndarray, std: np.ndarray):
	
	data = {
		"weights": weights,
		"biases": biases,
		"mean": mean.tolist(),
		"std": std.tolist()
	}

	file.write(json.dumps(data, indent=4))


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def read_weights(file: io.TextIOWrapper) -> WeightsFile:
	parsed = json.loads(file.read())

	return WeightsFile(
		weights=parsed["weights"],
		std=parsed["std"],
		mean=parsed["mean"],
		biases=parsed["biases"],
	)