import numpy as np
import matplotlib.pyplot as plt 
from random import shuffle

#load data
classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
label = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}

data = np.loadtxt('../data/iris.data', converters = {4: lambda x: classes[x]},
        delimiter=',')

X = data[:,0:4]
K = 20 # Tamanio del mapa 2D, neuronas de salida
N = X.shape[0]
features = X.shape[1]
w = np.random.rand(K,K,features)  #total de conexiones
plt.imshow(w, interpolation='none')
plt.savefig("w_initial.png")

def train_data(data):
	N = data.shape[0]
	features = data.shape[1]
	random_indices = np.arange(0,N)
	np.random.shuffle(random_indices)
	train_x = np.zeros((N,features))

	for i in range(0,N):
		indice = random_indices[i]
		train_x[i,:] += data[indice,:]
		
	return train_x


def best_matching_unit(x, w): # obtener la neurona ganadora
	dist = np.zeros((N,N))
	for i in range(0,K):
		for j in range(0,K):
			for k in range(0,features):
				dist[i,j] += (w[i,j,k] - x[k])**2   #Distancia euclidiana

	bmu = np.argmin(dist)  # Seleccionar el indice de menor valor
	return np.unravel_index(bmu,(K,K))  #coordenada de la neurona ganadora, unravel_index(): Convierte un indice plano o conjunto de indices planos en una tupla de coordenadas matrices.


def train(x_train, w, i):
    bmu = best_matching_unit(x_train[i],w)
    for x in range(0,K):
        for y in range(0,K):
            v = np.array([x,y])# vecinos en la coordenada x,y
            d = np.linalg.norm(v-bmu)  #distancia del minimo a sus vecinos
            L = learning_ratio(i)
            S = learning_radius(i,d)
            for z in range(0,features): 
                w[x,y,z] += L*S*(x_train[i,z] - w[x,y,z])  # Actualizando los pesos


def neighbourhood(t):# radio de los vecinos
    halflife = float(N/4) 
    initial  = float(K/2)
    return initial*np.exp(-t/halflife)

def learning_ratio(t):
    halflife = float(N/4) 
    initial  = 0.1
    return initial*np.exp(-t/halflife)

def learning_radius(t, d):
    s = neighbourhood(t)
    return np.exp(-d**2/(2*s**2))

# Entrenamiento
trainx = train_data(X)

max_iter = 500

for t in range(0, max_iter):
	for i in range(0,N):
	    train(trainx, w, i)

# Salida
plt.imshow(w, interpolation='none')
plt.savefig("w_final.png")