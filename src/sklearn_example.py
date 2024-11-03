
import pandas as pd
from pydantic import ConfigDict, validate_call
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


model_config = ConfigDict(arbitrary_types_allowed=True)

@validate_call(config=model_config)
def run(df: pd.DataFrame):
	df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)

	X = df.iloc[:, 2:]
	Y = df.iloc[:, 1]

	X_Train = X[:-50]
	Y_Train = Y[:-50]

	X_Test = X[-50:]
	Y_Test = Y[-50:]

	clf = MLPClassifier(hidden_layer_sizes=(20, 20),
						random_state=5,
						verbose=True,
						learning_rate_init=0.01)
	
	clf.fit(X_Train, Y_Train)

	ypred = clf.predict(X_Test)

	print(ypred, accuracy_score(Y_Test, ypred))