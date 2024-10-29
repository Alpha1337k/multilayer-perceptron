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

classifier_to_int = {
	'M': 0,
	'B': 1,
}

class Dataset(BaseModel):
	X: np.ndarray
	Y: np.ndarray

	XValidate: np.ndarray
	YValidate: np.ndarray

	@staticmethod
	def make(df: pd.DataFrame, validation_pct: int | None):
		# df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)

		df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: (x-x.mean())/x.std(), axis=0)


		X = df.iloc[:, 2:].to_numpy()
		Y = np.array([classifier_to_int[i] for i in df.iloc[:, 1].to_numpy()])

		if (validation_pct is not None):
			cutoff = int(len(X) * (validation_pct / 100))
		else:
			cutoff = None

		return Dataset(
			X = X[:-cutoff] if cutoff else X,
			XValidate = X[-cutoff:] if cutoff else X,
			Y = Y[:-cutoff] if cutoff else Y,
			YValidate = Y[-cutoff:] if cutoff else Y,
		)
