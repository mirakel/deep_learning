import numpy as np
import cPickle as pickle
import gzip
from PIL import Image
import matplotlib.pyplot as plt
import time
plt.ion()

def load_data():
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    	train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

def view_data(data):
	#de todos los datos de entrenamiento visualizar los 10 primeros
    for i in range(0,10):
    	plt.subplot(2, 5, i+1)
    	plt.imshow(data[i].reshape((28, 28)), cmap='gray')
    plt.show()

def view_layer_input(data):
	#visualizar el resultado del encode
	plt.title('Input Layer')
	plt.imshow(data.reshape((28, 28)), cmap='gray')
	plt.draw()
	time.sleep(0.3)

def view_layer_hidden(data):
	#visualizar el resultado del encode
	plt.title('Encode - Hidden Layer')
	plt.imshow(data.reshape((10, 10)), cmap='gray')
	plt.draw()
	time.sleep(0.3)

def view_layer_output(data):
	#visualizar el resultado de la capa de salida
    plt.title('Decode - Output Layer')
    plt.imshow(data.reshape((28, 28)), cmap='gray')
    plt.draw()
    time.sleep(0.3)

def visualize_weights(weights, panel_shape, tile_size):

    def scale(x):
        eps = 1e-8
        x = x.copy()
        x -= x.min()
        x *= 1.0 / (x.max() + eps)
        return 255.0*x

    margin_y = np.zeros(tile_size[1])
    margin_x = np.zeros((tile_size[0] + 1) * panel_shape[0])
    image = margin_x.copy()

    for y in range(panel_shape[1]):
        tmp = np.hstack( [ np.c_[ scale( x.reshape(tile_size) ), margin_y ] 
            for x in weights[y*panel_shape[0]:(y+1)*panel_shape[0]]])
        tmp = np.vstack([tmp, margin_x])

        image = np.vstack([image, tmp])

    img = Image.fromarray(image)
    img = img.convert('RGB')
    return img


#probando las funciones a las funciones
# train_set, valid_set, test_set  = load_data()
# train_x, train_y = train_set
# view_data(train_x)


