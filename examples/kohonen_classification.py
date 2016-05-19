import numpy as np
import matplotlib.pyplot as plt 
from random import shuffle

#load data
classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
label = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}

data = np.loadtxt('../data/iris.data', converters = {4: lambda x: classes[x]},
        delimiter=',')

def train_data(data, porcentaje):
	N = data.shape[0]
	features = data.shape[1]
	random_indices = np.arange(0,N)
	np.random.shuffle(random_indices)
	limit = N * porcentaje / 100.0
	train_x = np.zeros((limit,features))
	valid_x = np.zeros((N - limit,features))

	for i in range(0,N):
		indice = random_indices[i]
		if(i < limit):
			train_x[i,:] += data[indice,:]
		else:
			valid_x[i-limit,:] += data[indice,:]
	
	return train_x, valid_x

def best_matching_unit(x, w): # la neurona ganadora
	dist = np.sum((x_i - w) ** 2, axis=0)  #Distancia euclidiana
	bmu = np.min(dist)
	for i in range(0, len(dist)):
		if(dist[i] == bmu):   #posicion de la neurona ganadora
			return i  


train, valid = train_data(data,60)

trainx = train[:,0:4]
features = trainx.shape[1]
N = trainx.shape[0]
w = np.random.random((features,len(classes))) #inicializando los pesos
max_iter = 1000
rate = 1


#Entrenamiento
for t in range(1, max_iter):
	for i in range(0,N):
		x_i = trainx[i,:].reshape(features,1)
		winner = best_matching_unit(x_i, w)
		w[:,winner] = w[:,winner] + (rate/t) * (x_i.T - w[:,winner])  #actualizar los pesos solo de la neurona ganadora

print 'Los pesos encontrados:'
print w

#validacion 
M = valid.shape[0]
validx = valid[:,0:4]
validy = valid[:,-1]

for i in range(0,M):
	x_i = validx[i,:].reshape(features,1)
	winner = best_matching_unit(x_i, w)
	print 'Resultado:', label[winner], ' Objetivo: ', label[int(validy[i])]