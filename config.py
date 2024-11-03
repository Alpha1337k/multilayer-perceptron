Network(layers = [
	# in
	Layer(size=30, activation='sigmoid'),
	# hidden
	Layer(size=20, activation='sigmoid'),
	Layer(size=20, activation='sigmoid'),
	# out
	Layer(size=2, activation='softmax'),
])