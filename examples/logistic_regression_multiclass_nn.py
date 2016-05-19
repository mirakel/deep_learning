import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

#Generando la data
N = 200
A = 0.6*np.random.randn(N, 2)+[1, 1]
B = 0.6*np.random.randn(N, 2)+[3, 3]
C = 0.6*np.random.randn(N, 2)+[3, 0]
X = np.hstack((np.ones(3*N).reshape(3*N, 1), np.vstack((A, B, C))))
Y = np.vstack(((np.zeros(N)).reshape(N, 1),
        np.ones(N).reshape(N, 1), 2*np.ones(N).reshape(N, 1)))

w = np.zeros((3, 3))  #inicializando los pesos w0, w1
tol = 3E-1
rate = 1E-2
M = 3*N
it = 1;

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

while(True):
	grad_t = np.zeros((3, 3))
	J = 0
	for i in range(0,M):
		x = X[i,:]
		y = Y[i]
		z = np.dot(x,w) #salida de la red
		h_xi_w = softmax(z) #activacion de la red 
		J += -z[int(y)] + np.log(np.sum(np.exp(z))) #funcion costo 
		grad_t[:,int(y)] += x * (1 - h_xi_w[int(y)])
	
	J = (1.0 / M) * J   #costo total del conj de datos de entrenamiento
	w = w + (rate / M) * grad_t 
	print 'costo en la epoca',it, ': ', J
	it += 1
	if(J < tol): 
		break
	

print "pesos encontrados", w

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])

# Generate the mesh
x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
h = 0.02 # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_mesh = np.c_[np.ones((xx.size, 1)), xx.ravel(), yy.ravel()]
Z = np.zeros((xx.size, 1))

# Compute the likelihood of each cell in the mesh
for i in range(0, xx.size):
    lik = np.dot(X_mesh[i, :],w)
    Z[i] = np.argmax(lik)

# Plot it
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.plot(X[0:N-1, 1], X[0:N-1, 2], 'ro', X[N:2*N-1, 1], X[N:2*N-1,2], 'bo', X[2*N:, 1], X[2*N:, 2], 'go')
plt.axis([np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, np.min(X[:, 2])-0.5, np.max(X[:, 2])+0.5])
plt.savefig('multi_class.png')
plt.show()


