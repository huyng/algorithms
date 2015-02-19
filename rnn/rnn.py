"""
A vanilla RNN
"""

import numpy as np
import theano.tensor as T
import theano

n_u = 10             # input activations size
n_h = 10             # hidden activations size
n_y = 2              # output activation size
learning_rate = 0.5  
L1_reg = 0.5
L2_reg = 100
initial_momentum = 10
final_momentum = 1
momentum_switchover = 10
n_epochs = 100


x = T.matrix()
W_uh = theano.shared(value = np.asarray(np.random.uniform(
                                size = (n_u, n_h),
                                low = -.01, 
                                high = .01),
                             dtype = theano.config.floatX),
                    name = "W_uh")
W_hh = theano.shared(value = np.asarray(np.random.uniform(
                                size = (n_h, n_h),
                                low = -.01, 
                                high = .01),
                              dtype = theano.config.floatX),
                    name = "W_hh")
W_hy = theano.shared(value = np.asarray(np.random.uniform(
                                size = (n_h, n_y),
                                low = -.01, 
                                high = .01),
                             dtype = theano.config.floatX),
                    name = "W_hy")

# biases initialized to zeros
b_h = theano.shared(value=np.zeros((n_h,), dtype=theano.config.floatX))
b_y = theano.shared(value=np.zeros((n_h,), dtype=theano.config.floatX))


# set initial h0 activations to all zeros
h0 = theano.shared(value=np.zeros((n_h,), dtype= theano.config.floatX))

params = [W_uh, W_hh, W_hy, h0, b_h, b_y]

# set initial value for all gradient descent updates for all params
# to zeros

updates = {}
for param in params:
    pshape = param.get_value(borrow=True).shape
    updates[param] = theano.shared(value=np.zeros(pshape, 
                                                  dtype=theano.config.floatX), 
                                   name="updates")

def recurrent_fn(u_t, h_tm1):
    h_t = T.tanh(T.dot(u_t, W_uh) + T.dot(h_tm1, W_hh) + b_h)
    y_t = T.dot(h_t, W_hy) + b_y
    return (h_t, y_t)


(h, y_predict), _ = theano.scan(recurrent_fn, 
                                sequences = x,
                                outputs_info = [h0, None])


L1 = abs(W_uh.sum()) + abs(W_hh.sum()) + abs(W_hy.sum())
L2sq = (W_uh ** 2).sum() + (W_hh ** 2).sum() + (W_hy ** 2).sum()

# use mean squared error loss
loss = lambda y: T.mean(y_predict - y) ** 2
y = T.matrix(name = 'y', dtype = theano.config.floatX)
cost = loss(y) + L1_reg * L1 + L2_reg * L2sq
index = T.lscalar('index')
lr = T.scalar('lr', dtype = theano.config.floatX) 
mom = T.scalar('mom', dtype = theano.config.floatX) 

t = np.arange(300)
train_set_x = .5 * np.sin(t/2)
train_set_y = np.asarray(x[1:] + [1])

training_error = theano.function(inputs=[index],
                                 outputs = loss(y),
                                 givens = {
                                    x: train_set_x[index],
                                    y: train_set_y[index],
                                 },
                                 mode = "cvm")

param_grads = []
for param in params:
    param_grads.append(T.grad(cost, param))















