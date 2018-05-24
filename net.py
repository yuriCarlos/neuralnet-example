from neuron import Neuron
import numpy as np
from activation_func import d_sigmoide, sigmoid

class Net():
	def __init__(self):
		self.layers=[]
		self.g=sigmoid
		self.d_g=d_sigmoide

	#cria um layer com n neuronios
	def create_layer(self, n_num):
		self.layers.append([Neuron(None if len(self.layers) == 0 else self.layers[-1]) for i in range(0, n_num)])

	def predict(self, X):
		result=[]
		for x in X:
			result.append(self.feedforward(x))
		return result

	def feedforward(self, x):
		# entrada do dado na rede
		for idn, n in enumerate(self.layers[0]):
			n.out=x[idn]
			
		#passa os dados através dos layers
		for layer in self.layers:
			#print("resultados do layer")
			for n in layer:
				n.receive_data(self.g)
				#print(n.out)
		
		return [n.out for n in self.layers[-1]]
	
	def train(self, X, Y, learning_rate=.5, batch=1, epoch=50000, error_to_stop=.005):
		# contador de itens que passaram pelo batch
		batch_id = 0
		for i in range(0, epoch):
			# treina com essa epoca
			mean_square_error, batch_id = self._train(X, Y, learning_rate, batch_id, batch)
			print('EQM')
			print(mean_square_error)
			# se o erro for menor que o erro determinado de parada então para a rede
			if(mean_square_error <= error_to_stop):
				break

	def _train(self, X, Y, learning_rate, batch_id, batch_size):
		erro_quad_medio = .0
		for idx, x in enumerate(X):
			#alimenta a rede e pega o resultado
			self.feedforward(x)
			#calcula o erro de cada saida
			for nid, n in enumerate(self.layers[-1]):
				n.error = Y[idx][nid] - n.out
				erro_quad_medio += (Y[idx][nid] - n.out) * (Y[idx][nid] - n.out)
 
			#backpropaga o erro
			for i in reversed(range(0, len(self.layers))):
				for ndx, n in enumerate(self.layers[i]):
					n.delta = n.error * self.d_g(n.out)
					
					#calcula o delta w de cada aresta
					for w in n.weights:
						w.deltaw += learning_rate * n.delta * w.origin.out
						#acumula a contribuição de erro nos nós adjacentes
						w.origin.error += (n.delta * w.value)
					#zera o erro do no corrente para nao influenciar em calculos futuros
					n.error=0
				
			# atualiza os pesos da rede se chegou no tamanho do batch
			if batch_id == batch_size:
				for layer in self.layers:
					for n in layer:
						n.update_weights()
				
				batch_id = 0
		
			# atualiza o valor do batch
			batch_id += 1
			
		return erro_quad_medio/len(X), batch_id


net=Net()
net.create_layer(3)
net.create_layer(2)
net.create_layer(1)
print(net.layers)

X = [
	
	[.9,.9,.9],
	[.9,.9,.1],
	[.9,.1,.9],
	[.9,.1,.1]
]

Y=[
	[0],
	[1],
	[1],
	[0]
]

'''Y=[
	[1],
	[1],
	[1],
	[0]
]'''

net.train(X, Y, batch=8)
print(net.predict(X))
print(Y)