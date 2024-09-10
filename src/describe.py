import pandas as pd
from pydantic import ConfigDict, validate_call


model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=model_config)
def describe(df: pd.DataFrame):
	described = df.describe()

	for col in described:
		print(described[col], "\n")