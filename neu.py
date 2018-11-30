import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class Perceptron():

    def __init__(self, t_entrada, nodos):

        # Propiedades de la capa
        #self.nodos = nodos
        #self.entrada = t_entrada
        # Matriz de pesos
        self.W = np.squeeze(np.random.rand(t_entrada, nodos))
        self.b = np.random.rand(nodos)
        #self.func = func

    def predict(self, X):
        #print('W:', self.W)
        return sigmoid(np.dot(X, self.W) + self.b)

    def train(self, X, Y, alpha=0.1, verbose=0):
        # if func == 'lineal':
        Y_h = self.predict(X)
        #error = np.sum(np.squeeze(Y_h.T - Y)) / len(X)
        gradiente_b = (Y_h - Y) / len(X)
        gradiente_W = np.multiply(gradiente_b, X.T)
        #print(gradiente_W)
        self.W += -alpha * np.sum(gradiente_W, axis=1)
        self.b += -alpha * np.sum(gradiente_b)
        if verbose:
            print(np.sum(gradiente_b))
        return np.sum(gradiente_b)


def make_meshgrid(x, y, h=.002):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.squeeze(np.c_[xx.ravel(), yy.ravel()]))
    print(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, **params)


caracteristicas = 2
muestras = 1000  
centro1 = [0, 3]
centro2 = [0, 10]
X = np.random.randn(muestras, caracteristicas) + centro1
X = np.r_[X, np.random.randn(muestras, caracteristicas) + centro2]
X = X/10
Y = np.r_[np.zeros(muestras), np.ones(muestras)]
# Crear una capa de N nodos
N = 1
capa1 = Perceptron(caracteristicas, N)
salida = capa1.predict(X)
#print(salida)

h = []
for i in range(1000):
    error = capa1.train(X, Y, verbose=0)
    h.append(error)

Yf = capa1.predict(X)
print(Yf)
plt.plot(h)
plt.show()

xx, yy = make_meshgrid(X[:, 0], X[:, 1])

plot_contours(capa1, xx, yy,
              cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
plt.show()