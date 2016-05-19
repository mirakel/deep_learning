import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()

#load data
data = np.loadtxt('../data/hopfield_data.txt')
X = data[:,0:25]

N = X.shape[0]
M = X.shape[1] 
w = np.zeros((M,M))

def activacion(z):
	if z > 0:
		return 1
	else:
		return -1

## Vectorizamos la funcion para que evaluar arrays
activacion = np.vectorize(activacion)

#fase de almacenamiento: obtencion de pesos para los patrones de entrada
for i in range(0,N):
	x_i = X[i,:]
	w = w + np.outer(x_i,x_i) - np.identity(M)
	plt.title('Patron ingresado')
	plt.imshow(x_i.reshape((5, 5)), cmap='gray')
	plt.draw()
	time.sleep(0.8)
    

print 'pesos encontrados:', w


#fase de recuperacion: ingresar un patron con ruido y recuperar el mas parecido
#primer patron con ruido	
x_i = np.array([-1, -1, -1, -1, -1, 1, -1, -1, -1,-1, 1, -1, -1, -1, -1, 1,-1, -1, -1, -1, 1, -1, -1, 1, 1])

#Grafica
print 'xi',x_i
plt.title('Patron con ruido')
plt.imshow(x_i.reshape((5, 5)), cmap='gray')
plt.draw()
time.sleep(0.8)

#repetir el procedimiento hasta que sea estable, mientras que la salida sea distinta a la anterior s(t+1) <> s(t) 
while(True):	
	s_i = activacion(x_i.dot(w))
	#comparar las salidas  x_i y s_i
	if(np.array_equal(x_i, s_i)):
		break
	else:
		x_i = s_i

print 'Patron recuperado: ',s_i

#patron encontrado
plt.title('Patron encontrado')
plt.imshow(s_i.reshape((5, 5)), cmap='gray')
plt.draw()
time.sleep(0.8)




