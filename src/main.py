from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import argparse
from pydantic import ConfigDict, validate_call
import matplotlib.colors as mcolors
import numpy as np

import describe
import plot
import train

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
			describe.describe(df)
		case "plot":
			plot.plot(df)
		case "train":
			train.run(df)