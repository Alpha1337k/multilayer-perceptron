import json
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

from utils.layer import Layer

class BaseModel(PydanticBaseModel):
	class Config:
		arbitrary_types_allowed = True
		
class Network(BaseModel):
	layers: List[Layer]