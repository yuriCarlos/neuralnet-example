from neuron import Neuron
import numpy as np
from activation_func import d_sigmoide, sigmoid

class Net():
	def __init__(self, class_divisor=None):
		self.layers=[]
		self.g=sigmoid
		self.d_g=d_sigmoide
		self.class_divisor= class_divisor

	#cria um layer com n neuronios
	def create_layer(self, n_num):
		self.layers.append([Neuron(None if len(self.layers) == 0 else self.layers[-1]) for i in range(0, n_num)])

	def predict(self, X):
		result=[]
		for x in X:
			# get the predicted label
			result.append( [ int(predicted / self.class_divisor) for predicted in self.feedforward(x)] if self.class_divisor is not None else self.feedforward(x))
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
	
	def train(self, X, Y, learning_rate=.4, batch=1, epoch=50000, error_to_stop=.005):
		# contador de itens que passaram pelo batch
		batch_id = 0
		last_mean=10
		for i in range(0, epoch):
			print('epoch' + str(i))
			# treina com essa epoca
			mean_square_error, batch_id = self._train(X, Y, learning_rate, batch_id, batch)
			print('EQM')
			print(mean_square_error)
			# se o erro for menor que o erro determinado de parada então para a rede
			if(mean_square_error <= error_to_stop) or (last_mean - mean_square_error) < 0.00000001:
				break
			
			last_mean=mean_square_error


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


