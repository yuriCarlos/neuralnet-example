import math

def stair(value, thresh=.5):
	if(value > thresh):
		return 1
	return 0

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def d_sigmoide(x):
	return x * (1-x)