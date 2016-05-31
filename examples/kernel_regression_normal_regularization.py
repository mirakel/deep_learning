import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 
from matplotlib.legend_handler import HandlerLine2D

#Cargando los datos
X_file = np.genfromtxt('../data/ex5lin_ex2.csv', delimiter = ',', skip_header = 1)
N = np.shape(X_file)[0]  #cantidad de datos de entrenamiento
Y = X_file[:,1] 
X = X_file[:,0]
lambdaKernel = 0.8

def phi(x,u):
	return np.exp(-(1.0/lambdaKernel)*np.sum((x - u)**2))   #distancia euclidiana entre x y u

#Matriz PHI
Phi = np.zeros((N,N))

for i in range(0,N):
	for j in range(0,N):
		Phi[i,j] = phi(X[i], X[j])

features = Phi.shape[1]
delta = np.array([0,0.001,0.5,10]) #algunos deltas para la prueba
w = np.zeros((len(delta),features))

#aplicando la ecuacion normal para cada delta
for i in range(0,len(delta)):
	w[i,:] = np.dot(inv(np.dot(Phi.transpose(),Phi) + delta[i] * np.identity(features)), np.dot(Phi.transpose(),Y))
	#calculo del costo
	A = Y - np.dot(Phi,w[i,:])
	J = (np.dot(A.transpose(),A) + delta[i] * np.dot(w[i,:].transpose(),w[i,:]))/(2 * N)  #costo promedio

print "pesos encontrados: ", w
print "Costo: ", J
	
#Grafica
T = 20
test = np.linspace(np.min(X), np.max(X), T)
xx = np.zeros((N,T))

for i in range(0,N):
	for j in range(0,T):
		xx[i,j] = phi(X[i], test[j])


r_line = np.dot(xx.transpose(), w[0,:])
g_line = np.dot(xx.transpose(), w[1,:])
b_line = np.dot(xx.transpose(), w[2,:])
y_line = np.dot(xx.transpose(), w[3,:])

plt.title('Regresion Lineal con RBFs')
plt.plot(X, Y, 'kx', )
line1, = plt.plot(test, r_line, 'r-', label='delta = 0')
line2, = plt.plot(test, g_line, 'g--', label='delta = 0.001')
line3, = plt.plot(test, b_line, 'p-', label='delta = 0.5')
line4, = plt.plot(test, y_line, 'b:', label='delta = 10')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=5)})
plt.savefig('kernel_regularization.png')
plt.show()






