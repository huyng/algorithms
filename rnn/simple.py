"""
A vanilla RNN
"""

import numpy as np
import theano.tensor as T
import theano
# theano.config.exception_verbosity="high"


# ============
# Create model
# ============

n_x = 1  # input activations size for single time step
n_h = 10  # hidden activations size
n_y = 1   # output activation size


# inputs/outputs
# ==============

x = T.matrix(name='x', dtype=theano.config.floatX)
y = T.matrix(name='y', dtype=theano.config.floatX)


# parameters
# ==========

W_xh = theano.shared(value = np.asarray(np.random.uniform(
                                size = (n_x, n_h),
                                low = -.01, 
                                high = .01),
                             dtype = theano.config.floatX),
                    name = "W_xh")

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

b_h = theano.shared(value=np.zeros((n_h,), dtype=theano.config.floatX))
b_y = theano.shared(value=np.zeros((n_h,), dtype=theano.config.floatX))
h0 = theano.shared(value=np.zeros((n_h,), dtype= theano.config.floatX))


# training model
# ==============

alpha = T.scalar('alpha', dtype=theano.config.floatX)  # learning rate

def recurrent_fn(x_t, h_tm1):
    h_t = T.tanh(T.dot(x_t, W_xh) + T.dot(h_tm1, W_hh) + b_h)
    y_t = T.dot(h_t, W_hy) + b_y
    return (h_t, y_t)

(h, y_predict), _ = theano.scan(recurrent_fn, sequences=x, outputs_info=[h0, None])

loss =  T.mean(y_predict - y) ** 2
cost = loss

parameters = [W_xh, W_hh, W_hy, b_h, b_y, h0]

# symbolically generate the derivative of the cost w.r.t the parameters (AKA the gradient)
gradients = {}
for param in parameters:
    grad = T.grad(cost, param)
    gradients[param] = grad

param_updates = {}
for param in parameters:
    grad = gradients[param]
    param_updates[param] = param - alpha*grad


# create a function that performs a single gradient descent step on training data
def generate_learner(x_train, 
                    y_train,
                    param_updates):

    index = T.lscalar('index')
    learner_fn = theano.function(
            inputs=index,
            outputs=cost,
            updates=param_updates,
            givens={
                x: x_train[index],
                y: y_train[index]
            }
        )

    return learner_fn



def generate_training_data(timesteps, n_samples=10000):
    t = np.arange(timesteps)
    X = []
    Y = []
    for i in range(n_samples):
        sample = .5 * np.sin(t/2)
        X.append(sample.reshape(1, -1))
        Y.append(np.asarray(sample[1:]+[0]).reshape(1, -1))

    x_train = theano.shared(np.array(X), name="x_train")
    y_train = theano.shared(np.array(Y), name="x_train")
    return x_train, y_train



n_samples=10000
x_train, y_train = generate_training_data(300, n_samples=n_samples)
learner = generate_learner(x_train, y_train, param_updates)

def train(max_epochs=1000):
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        for idx in range(n_samples):
            example_cost = learner(idx)
            print example_cost


train()



# index = T.lscalar('index')
# alpha = T.scalar('lr', dtype=theano.config.floatX)  # learning rate

# set initial value for all gradient descent updates for all parameters to zeros

# updates = {}
# parameters = [W_xh, W_hh, W_hy, h0, b_h, b_y]
# for param in parameters:
#     pshape = param.get_value(borrow=True).shape
#     updates[param] = theano.shared(value=np.zeros(pshape, 
#                                                   dtype=theano.config.floatX), 
#                                    name="updates")


# # use mean squared error loss
# t = np.arange(300)
# X_train = []
# Y_train = []
# for i in range(10000):
#     sample = .5 * np.sin(t/2)
#     X_train.append(sample.reshape(-1, 1))
#     Y_train.append(np.asarray(sample[1:]).reshape(-1, 1))

# train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
# train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
# training_error_fn = theano.function(inputs=[index],
#                                     outputs = loss(y),
#                                     givens = {
#                                        x: train_set_x[index],
#                                        y: train_set_y[index],
#                                     })


# # generate gradient of cost function with respect to 
# # [W_xh, W_hh, W_hy, h0, b_h, b_y]

# param_grads = []
# for param in params:
#     param_grads.append(T.grad(cost, param))

# step_updates = {}
# for param, grad in zip(params, param_grads):
#     weight_update = updates[param]
#     upd = mom * weight_update - lr * grad
#     step_updates[weight_update] = upd
#     step_updates[param] = param + upd

# train_fn = theano.function(inputs=[index, lr, mom],
#                            outputs=cost,
#                            updates=step_updates,
#                            givens = {
#                                 x: train_set_x[index],
#                                 y: train_set_y[index],
#                            })


# print "Starting model training"
# learning_rate = 0.5  
# n_epochs = 100

# epoch = 0
# n_train = train_set_x.get_value(borrow=True).shape[0]

# while (epoch < n_epochs):
#     epoch += 1
#     for idx in range(n_train):
#         effective_momentum = final_momentum if epoch > momentum_switchover else initial_momentum
#         example_cost = train_fn(idx, learning_rate, effective_momentum)
#     train_losses = [training_error_fn(i) for i in range(n_train)]
#     epoch_avg_loss = np.mean(train_losses)
#     print("epoch:%i -- train loss %f -- lr: %f" % epoch, epoch_avg_loss, learning_rate)

#     learning_rate *= learning_rate_decay