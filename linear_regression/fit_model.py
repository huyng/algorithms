import numpy
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
rng = numpy.random


m = 100 # number of training examples
n = 1 # number of features
X = rng.randn(m, n)  # training input
Y = (X.squeeze() + rng.rand(m))*3 + 1  # training output
D = (X,Y)
training_epochs = 100

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(np.zeros(n), name="w")  # initialize weights randomly
b = theano.shared(0., name="b") # initialize biase to 0
h = T.dot(x, w) + b
cost = ((y - h)**2).sum()
gw = T.grad(cost=cost, wrt=w) 
gb = T.grad(cost=cost, wrt=b) 


# Compile
learning_rate = 0.1
train = theano.function(inputs=[x,y],
                        outputs=[h, cost],
                        updates=[ (w, w - learning_rate * gw),
                                  (b, b - learning_rate * gb)])
predict = theano.function(inputs=[x], outputs=h)


training_error = []

# stochastic gradient descent
for i in range(training_epochs):
    pred, err = train(X[i:i+1], Y[i:i+1])
    training_error.append(err)

plt.plot(training_error)
plt.figure()
plt.plot(X, predict(X).squeeze())
plt.scatter(X, Y)
plt.show()
raw_input()