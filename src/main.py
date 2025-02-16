from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import argparse
from pydantic import ConfigDict, validate_call
import matplotlib.colors as mcolors
import numpy as np

import describe
import plot
import predict
import sklearn_example
from split_dataset import split_dataset
import train

def load_df(input):
	df = pd.read_csv(args.input)

	labels = df['diagnosis'].copy()

	df.drop(labels=[ 'diagnosis' ], axis=1)

	return df

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="multilayer-perceptron",
		description="Classify cell nuclei"
	)

	subparser = parser.add_subparsers(dest="task")

	describe_parser = subparser.add_parser('describe')
	describe_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/train.csv')


	plot_parser = subparser.add_parser('plot')
	plot_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/train.csv')

	train_parser = subparser.add_parser('train')
	
	train_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/train.csv')
	train_parser.add_argument('-i', '--iterations', default='100', type=int)
	train_parser.add_argument('--learning-rate', default='0.5', type=float)
	train_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='output/weights.json')
	train_parser.add_argument('-c', '--config', type=argparse.FileType('r'), default='./config.py')


	split = subparser.add_parser('split')

	split.add_argument('-i', '--input', type=argparse.FileType('r'), default='data/raw.csv')
	split.add_argument('-v', '--validation-pct', type=int, default=20, choices=range(0, 100))
	split.add_argument('--train-path', type=str, default="./data/train.csv")
	split.add_argument('--validate-path', type=str, default="./data/validate.csv")

	test_parser = subparser.add_parser('train_test')
	test_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/train.csv')

	predict_parser = subparser.add_parser('predict')
	predict_parser.add_argument('-c', '--config', type=argparse.FileType('r'), default='./config.py')
	predict_parser.add_argument('--input', type=argparse.FileType('r'), default='./data/validate.csv')
	predict_parser.add_argument('-w', '--weights', type=argparse.FileType('r'), default='./output/weights.json')




	args = parser.parse_args()

	match args.task:
		case "describe":
			describe.describe(load_df(args.input))
		case "plot":
			plot.plot(load_df(args.input))
		case "split":
			split_dataset(args.input, args.validation_pct, args.train_path, args.validate_path)
		case "train":
			train.run(args.input, args.output, args.config, args.iterations, args.learning_rate)
		case "predict":
			predict.predict(args.weights, args.input, args.config)
		case "train_test":
			sklearn_example.run(load_df(args.input))
		case _:
			parser.print_help()
			exit(1)