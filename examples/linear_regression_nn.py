import numpy as np
import matplotlib.pyplot as plt 
import time
plt.ion()

#Cargando los datos
X_file = np.genfromtxt('../data/mpg.csv', delimiter = ',', skip_header = 1)
N = np.shape(X_file)[0]  #cantidad de datos de entrenamiento
X = np.hstack((np.ones(N).reshape(N,1), X_file[:,4].reshape(N,1)))  #formando un nuevo array con la columna de 1 y la cuarta columna del conj de entrenamiento
Y = X_file[:,0]  #Primera columna del conj. de entrenamiento

#Normalizando los datos para que la gradiente descente converga mucho mas rapido
X[:,1] = (X[:,1] - np.mean(X[:,1])) / np.std(X[:,1])

w = np.array([0, 0])  #inicializando los pesos w0, w1
grad_t = np.zeros((2, 1))  #matriz 2x1 de ceros 
max_iter = 150
rate = 1E-1

tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
bf_line = w[0]+w[1]*tt
plt.xlabel('Pesos (Normalizados)')
plt.ylabel('MPG')
plt.title('Regresion Lineal con Redes neuronales')	 
plt.plot(X[:, 1], Y, 'kx')
graph = plt.plot(tt, bf_line, 'r-')[0]


for t in range(0, max_iter):
	h_xi_w = np.dot(X,w.transpose())  #salida de la red
	grad_t = np.dot(X.transpose(),(h_xi_w - Y))
	J = (1.0 / (2 * N)) * np.sum((h_xi_w - Y) ** 2); 
	w = w - (rate / N) * grad_t 
	print 'costo iter',t, ': ', J
	#Graficando
	bf_line = w[0]+w[1]*tt 
	graph.set_ydata(bf_line)
	plt.draw()
	time.sleep(0.5)

print "pesos encontrados", w
plt.savefig('img.png')





