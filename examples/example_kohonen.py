import numpy as np
import matplotlib.pyplot as plt 

#load data
X = np.loadtxt('../data_kohonen.txt')

#inicializando los pesos
w = np.array([[0.4, 0.4, 0.5, 0.5],[0.5, 0.6, 0.6, 0.5]])


N = X.shape[0]
max_iter = 127

def menor_indice(dist):
	for i in range(0, len(dist)):
		if(dist[i] == np.min(dist)):
			return i

for t in range(1, max_iter):
	for i in range(0,N):
		x_i = X[i,:].reshape(2,1)
		d = np.sum((x_i - w) ** 2, axis=0)  #Distancia euclidiana 
		winner = menor_indice(d)
		w[:,winner] = w[:,winner] + (1.0/t) * (x_i.T - w[:,winner])  #actualizar los pesos solo de la neurona vencedora

	print 't: ', t
	print  w  

#validando 
pi = np.array([0.3, 0.8])
x_i = pi.reshape(2,1)
d = np.sum((x_i - w) ** 2, axis=0)  #Distancia euclidiana 
winner = menor_indice(d)
print winner


#graficando los w, los w viene a ser los centroides
plt.plot(X[:, 0], X[:,1], 'ro',w[0,:],w[1,:],'go')
plt.show()


#programacion del ejemplo http://thales.cica.es/rd/Recursos/rd98/TecInfo/07/capitulo6.html

