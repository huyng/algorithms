"""
A vanilla RNN
"""

import numpy as np
import theano.tensor as T
import theano
theano.config.exception_verbosity="high"

n_u = 10             # input activations size
n_h = 10             # hidden activations size
n_y = 2              # output activation size
learning_rate = 0.5  
learning_rate_decay = .999
L1_reg = 0.5
L2_reg = 100.0
initial_momentum = 10
final_momentum = 1
momentum_switchover = 10
n_epochs = 100


x = T.matrix(name = 'x', dtype = theano.config.floatX)
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
X_train = []
Y_train = []
for i in range(10000):
    sample = .5 * np.sin(t/2)
    X_train.append(sample.reshape(-1, 1))
    Y_train.append(np.asarray(sample[1:]).reshape(-1, 1))

train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
training_error_fn = theano.function(inputs=[index],
                                    outputs = loss(y),
                                    givens = {
                                       x: train_set_x[index],
                                       y: train_set_y[index],
                                    })


# generate gradient of cost function with respect to 
# [W_uh, W_hh, W_hy, h0, b_h, b_y]

param_grads = []
for param in params:
    param_grads.append(T.grad(cost, param))

step_updates = {}
for param, grad in zip(params, param_grads):
    weight_update = updates[param]
    upd = mom * weight_update - lr * grad
    step_updates[weight_update] = upd
    step_updates[param] = param + upd

train_fn = theano.function(inputs=[index, lr, mom],
                           outputs=cost,
                           updates=step_updates,
                           givens = {
                                x: train_set_x[index],
                                y: train_set_y[index],
                           })


print "Starting model training"
epoch = 0
n_train = train_set_x.get_value(borrow=True).shape[0]

while (epoch < n_epochs):
    epoch += 1
    for idx in range(n_train):
        effective_momentum = final_momentum if epoch > momentum_switchover else initial_momentum
        example_cost = train_fn(idx, learning_rate, effective_momentum)
    train_losses = [training_error_fn(i) for i in range(n_train)]
    epoch_avg_loss = np.mean(train_losses)
    print("epoch:%i -- train loss %f -- lr: %f" % epoch, epoch_avg_loss, learning_rate)

    learning_rate *= learning_rate_decay