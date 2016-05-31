import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 

#Cargando los datos
X_file = np.genfromtxt('../data/mpg.csv', delimiter = ',', skip_header = 1)
N = np.shape(X_file)[0]  #cantidad de datos de entrenamiento
X = np.hstack((np.ones(N).reshape(N,1), X_file[:,4].reshape(N,1)))  #formando un nuevo array con la columna de 1 y la cuarta columna del conj de entrenamiento
Y = X_file[:,0]  #Primera columna del conj. de entrenamiento

#Normalizando los datos para que la gradiente descente converga mucho mas rapido
X[:,1] = (X[:,1] - np.mean(X[:,1])) / np.std(X[:,1])

#calculo del los pesos con la ecuacion normal  w = (x.T x)-1 x.Ty
w = np.dot(inv(np.dot(X.transpose(),X)), np.dot(X.transpose(),Y)) 
#calculo del costo
A = Y - np.dot(X,w)
J = np.dot(A.transpose(),A) /(2 * N)  #costo promedio

print "pesos encontrados: ", w
print "Costo: ", J

#Grafica
tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
bf_line = w[0]+w[1]*tt 
plt.xlabel('Pesos (Normalizados)')
plt.ylabel('MPG')
plt.title('Regresion Lineal con la Ecuacion Normal')	 
plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-' )
plt.savefig('img_normal.png')
plt.show()






