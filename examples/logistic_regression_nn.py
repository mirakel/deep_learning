import numpy as np
import matplotlib.pyplot as plt 
import time
plt.ion()

#Generando la data
N = 100
A = 0.3*np.random.randn(N, 2)+[1, 1]
B = 0.3*np.random.randn(N, 2)+[3, 3]
X = np.hstack((np.ones(2*N).reshape(2*N, 1), np.vstack((A, B))))
Y = np.vstack(((np.zeros(N)).reshape(N, 1), np.ones(N).reshape(N, 1)))

w = np.zeros((3, 1))  #inicializando los pesos w0, w1
max_iter = 200
rate = 5E-1
M = 2*N
grad_thresh = 5

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

tt = np.linspace(np.min(X[:, 1])-1, np.max(X[:, 1])+1, 10)
line = np.zeros(10)  

plt.plot(X[0:N-1, 1], X[0:N-1, 2], 'ro', X[N:, 1], X[N:, 2], 'bo')
graph = plt.plot(tt, line, 'k-')[0]

plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.axis([np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, np.min(X[:, 2])-0.5, np.max(X[:, 2])+0.5])

for t in range(0, max_iter):
	grad_t = np.zeros((3, 1))  #matriz 3x1 de ceros 
	h_xi_w = sigmoid(np.dot(X,w))  #salida de la red
	J = (1.0 / M) * (- np.dot(Y.transpose(),np.log(h_xi_w)) - np.dot((1 - Y).transpose(),(np.log(1 - h_xi_w)))) #funcion costo 
	grad_t = np.dot(X.transpose(),(h_xi_w - Y)) 
	w = w - (rate / M) * grad_t 
	print 'costo iter',t, ': ', J

	grad_norm = np.linalg.norm(grad_t)
	if grad_norm < grad_thresh:
		print "Converge en ", t+1," pasos."
		break

	#Graficando
	line = -w[0,0]/w[2,0]-w[1,0]/w[2,0]*tt #Linea de frontera
	graph.set_ydata(line)
	plt.draw()
	time.sleep(0.5)

print "pesos encontrados", w

plt.fill_between(tt, -1, line, facecolor='red', alpha=0.5)
plt.fill_between(tt, line, 5, facecolor='blue', alpha=0.5)
plt.savefig('two_class.png')
