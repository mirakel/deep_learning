import numpy as np
import matplotlib.pyplot as plt 

#Cargando los datos
X_file = np.genfromtxt('../data/mpg.csv', delimiter = ',', skip_header = 1)
N = np.shape(X_file)[0]  #cantidad de datos de entrenamiento
X = np.hstack((np.ones(N).reshape(N,1), X_file[:,4].reshape(N,1)))  #formando un nuevo array con la columna de 1 y la cuarta columna del conj de entrenamiento
Y = X_file[:,0]  #Primera columna del conj. de entrenamiento

#Normalizando los datos para que la gradiente descente converga mucho mas rapido
X[:,1] = (X[:,1] - np.mean(X[:,1])) / np.std(X[:,1])

w = np.array([0, 0])  #inicializando los pesos w0, w1

max_iter = 100
rate = 1E-3

for t in range(0, max_iter):
	grad_t = np.array([0.,0.])
	for i in range(0,N):
		x_i = X[i,:]
		y_i = Y[i]
		h_xi_w = np.dot(w,x_i)  #calculando la salida de la red
		grad_t += 2 * x_i * (h_xi_w - y_i)  #gradiente

	w = w - rate * grad_t 

print "pesos encontrados", w

tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
print tt
bf_line = w[0]+w[1]*tt 
print 'recta',bf_line

plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-')
plt.xlabel('Pesos (Normalizados)')
plt.ylabel('MPG')
plt.title('Regresion Lineal con Redes neuronales')
 
plt.savefig('mpg.png')
 
plt.show()