from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import argparse
from pydantic import ConfigDict, validate_call
import matplotlib.colors as mcolors
import numpy as np

import describe
import plot
import sklearn_example
from split_dataset import split_dataset
import train

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="multilayer-perceptron",
		description="Classify cell nuclei"
	)

	subparser = parser.add_subparsers(dest="task")

	subparser.add_parser('describe')
	subparser.add_parser('plot')
	subparser.add_parser('train')
	split = subparser.add_parser('split')

	split.add_argument('-i', '--input', type=argparse.FileType('r'), default='data/raw.csv')
	split.add_argument('-v', '--validation-pct', type=int, default=20, choices=range(0, 100))
	split.add_argument('--train-path', type=str, default="./data/train.csv")
	split.add_argument('--validate-path', type=str, default="./data/validate.csv")

	subparser.add_parser('train_test')

	args = parser.parse_args()

	# df = pd.read_csv(args.input, names=["id", "diagnosis"] + [f"c_{i}" for i in range(0, 30)] )

	# labels = df['diagnosis'].copy()

	# df.drop(labels=[ 'diagnosis' ], axis=1)

	match args.task:
		case "describe":
			describe.describe(df)
		case "plot":
			plot.plot(df)
		case "split":
			split_dataset(args.input, args.validation_pct, args.train_path, args.validate_path)
		case "train":
			train.run(df)
		case "train_test":
			sklearn_example.run(df)
		case _:
			parser.print_help()
			exit(1)