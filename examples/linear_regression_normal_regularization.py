import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 
from matplotlib.legend_handler import HandlerLine2D

#Cargando los datos
X_file = np.genfromtxt('../data/ex5lin_ex2.csv', delimiter = ',', skip_header = 1)
N = np.shape(X_file)[0]  #cantidad de datos de entrenamiento
Y = X_file[:,1]  #Primera columna del conj. de entrenamiento

# Hipotesis con un polinomio de grado 5, entonces modificar los datos de entrada para que acepte 5 parametros w
X = np.hstack((np.ones(N).reshape(N,1), X_file[:,0].reshape(N,1), (X_file[:,0]**2).reshape(N,1),(X_file[:,0]**3).reshape(N,1),(X_file[:,0]**4).reshape(N,1),(X_file[:,0]**5).reshape(N,1))) 
features = X.shape[1]
delta = np.array([0,1,8,10]) #algunos deltas para la prueba
w = np.zeros((len(delta),features))

#aplicando la ecuacion normal para cada delta
for i in range(0,len(delta)):
	w[i,:] = np.dot(inv(np.dot(X.transpose(),X) + delta[i] * np.identity(features)), np.dot(X.transpose(),Y))
	#calculo del costo
	A = Y - np.dot(X,w[i,:])
	J = (np.dot(A.transpose(),A) + delta[i] * np.dot(w[i,:].transpose(),w[i,:]))/(2 * N)  #costo promedio

print "pesos encontrados: ", w
print "Costo: ", J
	
#Grafica
tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
T = len(tt)
xx = np.hstack((np.ones(T).reshape(T,1), tt.reshape(T,1), (tt**2).reshape(T,1), (tt**3).reshape(T,1), (tt**4).reshape(T,1), (tt**5).reshape(T,1)))

r_line = np.sum(w[0,:] * xx, axis = 1)
g_line = np.sum(w[1,:] * xx, axis = 1)
b_line = np.sum(w[2,:] * xx, axis = 1)
y_line = np.sum(w[3,:] * xx, axis = 1)

plt.title('Regresion Lineal con Regularizacion')
plt.plot(X[:, 1], Y, 'kx', )
line1, = plt.plot(tt, r_line, 'r-', label='delta = 0')
line2, = plt.plot(tt, g_line, 'g--', label='delta = 1')
line3, = plt.plot(tt, b_line, 'b:', label='delta = 8')
line4, = plt.plot(tt, y_line, 'p-', label='delta = 10')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=5)})
plt.savefig('img_regularization.png')
plt.show()






