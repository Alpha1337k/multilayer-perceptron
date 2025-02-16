from typing import IO
import pandas as pd
from pydantic import validate_call


@validate_call
def split_dataset(input, validation_pct: int, train_path: str, validate_path: str):
	header = ['index', 'diagnosis'] + [f"c_{x}" for x in range(30)]

	df = pd.read_csv(input, header=None)

	cutoff = int(len(df) * (validation_pct / 100))

	train = df[:-cutoff]
	validate = df[-cutoff:]

	train.to_csv(train_path, header=header, index=False)
	validate.to_csv(validate_path, header=header, index=False)

	print(f"Training dataset saved to {train_path} ({len(train)} rows)")
	print(f"Validation dataset saved to {validate_path} ({len(validate)} rows)")