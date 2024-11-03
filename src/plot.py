
from typing import List
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call

model_config = ConfigDict(arbitrary_types_allowed=True)

@validate_call
def plot_log_loss(train: List[float], validate: List[float]):
	plt.plot(train, 'blue')
	plt.plot(validate, 'orange')
	plt.title('log loss')

	plt.savefig('output/training/log_loss.png')
	plt.clf()

@validate_call
def plot_accuracy(train: List[float], validate: List[float]):
	plt.plot(train, 'blue')
	plt.plot(validate, 'orange')
	plt.title('Accuracy')

	plt.savefig('output/training/accuracy.png')
	plt.clf()

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