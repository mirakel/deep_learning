import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()

#load data
data = np.loadtxt('../data/numbers.txt')

def activacion(z):
	if z > 0:
		return 1
	else:
		return -1

def ceros_a_negativos(z):
	if z <= 0:
		return -1
	else:
		return 1

## Vectorizamos la funcion para que evaluar arrays
activacion = np.vectorize(activacion)
ceros_a_negativos = np.vectorize(ceros_a_negativos)

X = ceros_a_negativos(data[:,0:120])
N = X.shape[0]
M = X.shape[1] 
w = np.zeros((M,M))

#fase de almacenamiento: obtencion de pesos para los patrones de entrada
for i in range(0,N):
	x_i = X[i,:]
	w = w + np.outer(x_i,x_i) - np.identity(M)
	plt.title('Patron ingresado')
	plt.imshow(x_i.reshape((10, 12)), cmap='gray')
	plt.draw()
	time.sleep(0.8)

#fase de recuperacion: ingresar un patron con ruido y recuperar el mas parecido
for i in range(0,N - 3):  #probando con algunos patrones
	x_i = X[i,:]
	noise = 0.3
	x_i = np.random.binomial(size = x_i.shape[0], n = 1, p = 1.0 - noise) * x_i  #filtro binomial
	plt.title('Patron con ruido')
	plt.imshow(x_i.reshape((10, 12)), cmap='gray')
	plt.draw()
	time.sleep(0.3)

	#repetir el procedimiento hasta que sea estable, mientras que la salida sea distinta a la anterior s(t+1) <> s(t) 
	while(True):	
		s_i = activacion(x_i.dot(w))
		if(np.array_equal(x_i, s_i)): #comparar las salidas  x_i y s_i
			break
		else:
			x_i = s_i

	print 'Patron encontrado: ',s_i
	plt.title('Patron encontrado')
	plt.imshow(s_i.reshape((10, 12)), cmap='gray')
	plt.draw()
	time.sleep(0.3)

