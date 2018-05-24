from random import randint
INDICE=0

class Neuron():
	def __init__(self, layer_to_connect):
		self.out=.0
		self.delta=.0
		self.error=0
		self.weights=[]
		global INDICE

		self.id=INDICE
		INDICE+=1

		#conecta este neuronio com todos os neuronios desta camada
		if layer_to_connect is not None:
			for n in layer_to_connect:
				self.weights.append(Weight(n))

	def update_weights(self):
		#print('delta: ')
		for w in self.weights:
			w.update()
		#print('delta================')

	def receive_data(self, ativation_func=None):
		out = .0
		for w in self.weights:
			out = out + w.value * w.origin.out

		#print('+++++++++++++++++++++++++++ SAIDA REAL')
		#print(out)
		
		#se for da primeira camada n aplica função de ativação
		if len(self.weights) != 0:
			self.out = out if ativation_func is None else ativation_func(out)
			#print('out = '+str(self.out))

class Weight():
	def __init__(self, n):
		self.value=float(randint(1,200))/100-1
		self.origin=n
		self.deltaw=0
	
	def update(self):
		#print('Old value: '+str(self.value))
		self.value=self.value + self.deltaw
		#print('New value: '+str(self.value))
		self.deltaw=0