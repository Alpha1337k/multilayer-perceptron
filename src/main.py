from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import argparse
from pydantic import ConfigDict, validate_call
import matplotlib.colors as mcolors
import numpy as np

model_config = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=model_config)
def describe(df: pd.DataFrame):
	described = df.describe()

	for col in described:
		print(described[col], "\n")

@validate_call(config=model_config)
def plot_dataset(path: str, b: pd.Series, m: pd.Series):
	plt.plot()

	plt.hist(b, color='red', bins=50, weights=np.ones(len(b)) / len(b))
	plt.hist(m, color='blue', bins=50, weights=np.ones(len(m)) / len(m))

	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

	plt.title(path)
	plt.savefig(path)
	plt.clf()

@validate_call(config=model_config)
def plot(df: pd.DataFrame):
	df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)



	data_M = df[df['diagnosis'] == 'M']
	data_B = df[df['diagnosis'] == 'B']

	# print(data_B)

	for col in data_B:
		plot_dataset(f"output/plot/b_{col}.png", data_B[col], data_M[col])

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

	df.drop(labels=[ 'diagnosis' ], axis=1)

	match args.task:
		case "describe":
			describe(df)
		case "plot":
			plot(df)
		