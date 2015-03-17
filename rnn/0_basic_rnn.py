"""
A vanilla RNN
"""

import numpy as np
import theano.tensor as T
import theano
import datasets
from numpy.random import uniform
from theano import config

# ======
# Config
# ======

# theano.config.exception_verbosity="high"
# theano.config.optimizer='fast_compile'
# theano.config.compute_test_value='off'



# ============
# Create model
# ============
n_x = 1   # input activations size for a single time step
n_h = 10  # hidden activations size for a single time step
n_y = 1   # output activation size for a single time step


# inputs/outputs
# ==============
x = T.matrix(name='x')
y = T.matrix(name='y')


# parameters
# ==========

# concrete values
W_xh_val = np.asarray(uniform(size=(n_x, n_h), low=-.01, high=.01), dtype=config.floatX)
W_hh_val = np.asarray(uniform(size=(n_h, n_h), low=-.01, high=.01), dtype=config.floatX)
W_hy_val = np.asarray(uniform(size=(n_h, n_y), low=-.01, high=.01), dtype=config.floatX)
b_h_val = np.zeros((n_h,), dtype=config.floatX)
b_y_val = np.zeros((n_y,), dtype=config.floatX)
h0_val = np.zeros((n_h,), dtype=config.floatX)

# symbolic
W_xh = theano.shared(value=W_xh_val, name="W_xh")
W_hh = theano.shared(value=W_hh_val, name="W_hh")
W_hy = theano.shared(value=W_hy_val, name="W_hy")
b_h = theano.shared(value=b_h_val, name="b_h")
b_y = theano.shared(value=b_y_val, name="b_y")
h0 = theano.shared(value=h0_val, name="h0")

parameters = [W_xh, W_hh, W_hy, b_h, b_y, h0]



def step(x_t, h_tm1):
    """
    This is a function representing a single time step of the 
    recurrent neural net
    """

    g_t = T.dot(x_t, W_xh) + T.dot(h_tm1, W_hh) + b_h
    h_t = T.tanh(g_t)
    y_t = T.dot(h_t, W_hy) + b_y
    return (h_t, y_t)

[h, y_predict], _ = theano.scan(step, sequences=x, outputs_info=[h0, None])

lr = T.scalar('lr', dtype=theano.config.floatX)
loss =  T.mean((y_predict - y) ** 2)
cost = loss


# symbolically generate the derivative of the cost
# w.r.t the parameters (AKA the gradient) and the 
# updates for each parameter
param_updates = []
for param in parameters:
    gradient = T.grad(cost, param)
    update = param - lr*gradient
    param_updates.append((param, update))


# create a function that performs a single gradient descent step on training data
def make_learner(x_train, y_train, param_updates):

    index = T.lscalar('index')
    learner_fn = theano.function(
            inputs=[index, lr],
            outputs=[cost],
            updates=param_updates,
            givens={
                x: x_train[index],
                y: y_train[index]
            }
        )

    return learner_fn

def make_predictor():
    predict_fn = theano.function(
            inputs=[x],
            outputs=[y_predict],
        )
    return predict_fn



def train(x_train_val, y_train_val):
    x_train = theano.shared(x_train_val)
    y_train = theano.shared(y_train_val)
    n_samples = x_train_val.shape[0]

    # creating training function
    learn_fn = make_learner(x_train, y_train, param_updates)

    # start gradient descent w/ batch_size==1
    max_epochs = 1000
    lr_val = 0.5
    for epoch in range(max_epochs):
        costs = []
        for idx in range(n_samples):
            sample_cost = learn_fn(idx, lr_val)
            costs.append(sample_cost[0])

        print "epoch=%-6s -- cost=%0.8f" % (epoch, np.mean(costs))



if __name__ == '__main__':
    # generate some simple training data
    n_samples = 100
    timesteps = 20
    x_train_val, y_train_val = datasets.sinewaves(timesteps, n=n_samples)
    train(x_train_val, y_train_val)
    predictor = make_predictor()

