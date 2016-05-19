import numpy as np
import matplotlib.pyplot as plt 

X = np.array([[1,1,-1,-1],[-1,-1,1,1]])
N = X.shape[0]
M = X.shape[1]
w = np.zeros((M,M))

#fase de almacenamiento: obtencion de pesos para los patrones de entrada

for i in range(0,N):
	x_i = X[i,:]
	w = w + np.outer(x_i,x_i) - np.identity(M)

print 'pesos encontrados:'
print w

#fase de recuperacion: elegir un patron y encontrar el mas parecido
x_i = np.array([1,-1,-1,-1])

#repetir el procedimiento hasta que sea estable, mientras que la salida sea distinta a la anterior s(t+1) <> s(t) 
while(True):	
	s_i = np.sign(x_i.dot(w))
	#comparar las salidas  x_i y s_i
	if(np.array_equal(x_i, s_i)):
		break
	else:
		x_i = s_i

print 'Patron encontrado: ',s_i


