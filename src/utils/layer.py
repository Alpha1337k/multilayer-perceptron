import math
import numpy as np
import pandas as pd
from typing import List, Optional, Any, Tuple
from pydantic import (
    BaseModel as PydanticBaseModel,
    ConfigDict,
    Field,
    FilePath,
    field_validator,
    model_validator,
    validate_call,
)

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
    
class Layer(BaseModel):
	size: int
	activation: str

	def sigmoid(self, s: float):
		return 1 / math.e ** -(s)

	def softmax(self, s: float):
		return s / math.e ** -(s)

	def activate(self, s: float):
		match self.activation:
			case "sigmoid":
				return self.sigmoid(s)
			case "softmax":
				return self.softmax(s)
		raise Exception("Activation not found.")
				