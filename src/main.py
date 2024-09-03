import pandas as pd
import argparse
from pydantic import ConfigDict, validate_call

model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=model_config)
def describe(df: pd.DataFrame):
	described = df.describe()

	for col in described:
		print(described[col], "\n")

@validate_call(config=model_config)
def plot(df: pd.DataFrame):
	data_M = df[df['diagnosis'] == 'M']
	data_B = df[df['diagnosis'] == 'B']



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="multilayer-perceptron",
		description="Classify cell nuclei"
	)

	parser.add_argument('-i', '--input', type=argparse.FileType('r'), default='data/raw.csv')

	subparser = parser.add_subparsers(dest="task")

	subparser.add_parser('describe')
	subparser.add_parser('plot')
	subparser.add_parser('train')

	args = parser.parse_args()

	df = pd.read_csv(args.input, names=["id", "diagnosis"] + [f"c_{i}" for i in range(0, 30)] )

	labels = df['diagnosis'].copy()

	# df.drop(labels=['id', 'diagnosis'], axis=1)

	match args.task:
		case "describe":
			describe(df)
		case "plot":
			plot(df, labels)
		