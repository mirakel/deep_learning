import numpy as np
import argparse
import cPickle as pickle
import utils

class Autoencoder(object):
	def __init__(self, n_visible = 784, n_hidden = 784, \
        W1 = None, W2 = None, b1 = None, b2 = None, 
        noise = 0.0):

		r = np.sqrt(6.0 / ( n_hidden + n_visible + 1))

		#inicializando los pesos
		if W1 == None:
		    self.W1 = self.random_init(r, (n_hidden, n_visible))  #funcion para muestras uniformes

		if W2 == None:
			self.W2 = self.random_init(r, (n_visible, n_hidden))

		if b1 == None:
		    self.b1 = np.zeros(n_hidden)

		if b2 == None:
		    self.b2 = np.zeros(n_visible)

		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.alpha = 0.1
		self.noise = noise

	def random_init(self, r, size):
		return np.array(np.random.uniform(low = -r, high = r, size=size))  #generando la data

   	def sigmoid(self, x):
   		return 1.0 / (1.0 + np.exp(-x))

   	def sigmoid_prime(self, x):
   		return x * (1.0 - x)

   	def corrupt(self, x, noise):  #agregando ruido a la entrada
   		return np.random.binomial(size = x.shape, n = 1, p = 1.0 - noise) * x

   	def encode(self, x):
   		return self.sigmoid(np.dot(self.W1, x) + self.b1)

   	def decode(self, y):
   		return self.sigmoid(np.dot(self.W2, y) + self.b2)

   	def cost(self, y, h):  #funcion de costo o de error
   		return - np.sum((y * np.log(h) + (1.0 - y) * np.log(1.0 - h)))

   	def feedforward(self, x):
		if(self.noise != 0.0):  #agregar ruido a los datos
			x = self.corrupt(x, self.noise)	
		    
		p = self.encode(x)
		h = self.decode(p)

		return p,h

	def backpropagation(self,x_batch):
		cost = 0.0
		grad_W1 = np.zeros(self.W1.shape)
		grad_W2 = np.zeros(self.W2.shape)
		grad_b1 = np.zeros(self.b1.shape)
		grad_b2 = np.zeros(self.b2.shape)

		for x in x_batch:
			p, h = self.feedforward(x)
			cost += self.cost(x,h)
			delta2 = - (x - h)   #delta de la capa de salida
			grad_W2 += np.outer(delta2, p)  #producto exterior entre 2 vectores
			grad_b2 += delta2
			delta1 = np.dot(self.W2.T, delta2) * self.sigmoid_prime(p) #delta de la capa oculta
			grad_W1 += np.outer(delta1, x)  
			grad_b1 += delta1   

		cost /= len(x_batch)  #promedio del minibatch
		grad_W1 /= len(x_batch)
		grad_W2 /= len(x_batch)
		grad_b1 /= len(x_batch)
		grad_b2 /= len(x_batch)

		#actualizando los pesos
		self.W1 = self.W1 - self.alpha * grad_W1
		self.W2 = self.W2 - self.alpha * grad_W2
		self.b1 = self.b1 - self.alpha * grad_b1
		self.b2 = self.b2 - self.alpha * grad_b2

		return cost

	def train(self, X, epochs = 15, batch_size = 20):
		batch_num = len(X) / batch_size

		for epoch in range(epochs): 
		    total_cost = 0.0
		    for i in range(batch_num):
		        batch = X[i*batch_size : (i+1)*batch_size]
		        cost = self.backpropagation(batch)
		        total_cost += cost
		       
		    print 'Epoca: ', epoch
		    print 'Error total: ',(1.0 / batch_num) * total_cost

	def validation(self, valid_set):
		for x in valid_set:  #para cada dato de validacion
			p = self.encode(x)
			h = self.decode(p)

			#visualizando la capa de entrada
			utils.view_layer_input(x)
			
			#visualizando la capa oculta
			utils.view_layer_hidden(p)

			#visualizando la capa de salida
			utils.view_layer_output(h)


	def dump_weights(self, save_path):  #funcion para guardar los pesos despues del entrenamiento
		with open(save_path, 'w') as f:
		    d = {
		        "W1" : self.W1,
		        "W2" : self.W2,
		        "b1" : self.b1,
		        "b2" : self.b2,
		        }

		    pickle.dump(d, f) 


	def visualize_weights(self):
		tile_size = (int(np.sqrt(self.W1[0].size)), int(np.sqrt(self.W1[0].size)))
		panel_shape = (10, 10)
		return utils.visualize_weights(self.W1, panel_shape, tile_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_hidden", type = int)
    parser.add_argument("-e", "--epochs", type = int, default = 1)
    parser.add_argument("-b", "--batch_size", type = int, default = 20)
    parser.add_argument("-n", "--noise", type=float, default = 0.0)
    parser.add_argument('-o', '--output', type = unicode)
    parser.add_argument('-v', '--visualize', action = "store_true")  #false
    args = parser.parse_args()

    train_data, valid_data, test_data = utils.load_data()

    autoencoder = Autoencoder(n_hidden = args.n_hidden, noise = args.noise)

    train_x, train_y = train_data
    valid_x, valid_y = valid_data

    try:
        autoencoder.train(train_x, epochs = args.epochs, batch_size = args.batch_size)
        autoencoder.validation(valid_x[0:20,:])  #validando con 20 datos
    except KeyboardInterrupt:
        exit()
        pass

    save_name = args.output

    if save_name == None:
        save_name = 'visualize_weigth'

    img = autoencoder.visualize_weights()
    img.save(save_name + ".bmp")

    if args.visualize:    
        img.show()

    autoencoder.dump_weights(save_name + '.pkl')