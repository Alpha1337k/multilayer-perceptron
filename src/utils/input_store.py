

from typing import Dict, List

import numpy as np
from pydantic import Field, validate_call
from utils.layer import BaseModel
from utils.network import Network


class InputStore(BaseModel):
	inputs: List[float]
	network: Network

	cached_inputs: List[List[float]] = []
	is_init: bool = False

	def init(self):
		self.is_init = True
		self.cached_inputs = [[0] * layer.size for layer in self.network.layers]

	def get_output(self, layer: int, node: int):
		return self.cached_inputs[layer][node]

	@validate_call
	def get_inputs(self, layer: int, node: int):
		if (layer == 0):
			return [self.inputs[node]]
		if (self.is_init == False):
			self.init()
		return self.cached_inputs[layer - 1]
	
	def get_layer(self, layer: int) -> List[float]:
		return self.cached_inputs[layer - 1]
		

	@validate_call
	def set_inputs(self, layer: int, node: int, val: float):
		if (self.is_init == False):
			self.init()
		self.cached_inputs[layer][node] = val