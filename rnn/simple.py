"""
A vanilla RNN
"""

import numpy as np
import theano.tensor as T
import theano

# theano.config.exception_verbosity="high"
# theano.config.optimizer='fast_compile'
# theano.config.compute_test_value='off'

# ============
# Create model
# ============

n_x = 1   # input activations size for a single time step
n_h = 10  # hidden activations size
n_y = 1   # output activation size


# inputs/outputs
# ==============

x = T.matrix(name='x', dtype=theano.config.floatX)
y = T.matrix(name='y', dtype=theano.config.floatX)


# parameters
# ==========

# concrete values
W_xh_val = np.asarray(np.random.uniform(size=(n_x, n_h), low=-.01, high=.01), dtype=theano.config.floatX)
W_hh_val = np.asarray(np.random.uniform(size=(n_h, n_h), low=-.01, high=.01), dtype=theano.config.floatX)
W_hy_val = np.asarray(np.random.uniform(size=(n_h, n_y), low=-.01, high=.01), dtype=theano.config.floatX)
b_h_val = np.zeros((n_h,), dtype=theano.config.floatX)
b_y_val = np.zeros((n_y,), dtype=theano.config.floatX)
h0_val = np.zeros((n_h,), dtype= theano.config.floatX)

# symbolic
W_xh = theano.shared(value=W_xh_val, name="W_xh")
W_hh = theano.shared(value=W_hh_val, name="W_hh")
W_hy = theano.shared(value=W_hy_val, name="W_hy")
b_h = theano.shared(value=b_h_val, name="b_h")
b_y = theano.shared(value=b_y_val, name="b_y")
h0 = theano.shared(value=h0_val, name="h0")

parameters = [W_xh, W_hh, W_hy, b_h, b_y, h0]


# training model
# ==============
def recurrent_fn(x_t, h_tm1):
    h_t = T.tanh(T.dot(x_t, W_xh) + T.dot(h_tm1, W_hh) + b_h)
    y_t = T.dot(h_t, W_hy) + b_y
    return (h_t, y_t)

(h, y_predict), _ = theano.scan(recurrent_fn, sequences=x, outputs_info=[h0, None])

loss =  T.mean(y_predict - y) ** 2
cost = loss
alpha = T.scalar('alpha', dtype=theano.config.floatX)  # learning rate


# symbolically generate the derivative of the cost w.r.t the parameters (AKA the gradient)
gradients = {}
for param in parameters:
    grad = T.grad(cost, param)
    gradients[param] = grad

param_updates = []
for param in parameters:
    grad = gradients[param]
    update = param - alpha*grad
    param_updates.append((param, update))


# create a function that performs a single gradient descent step on training data
def generate_learner(x_train, y_train, param_updates):

    index = T.lscalar('index')
    learner_fn = theano.function(
            inputs=[index, alpha],
            outputs=[cost],
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
        sample_input = .5 * np.sin(t/(2*i))
        sample_output = np.roll(sample_input, -1)
        X.append(sample_input.reshape(-1, 1))
        Y.append(sample_output.reshape(-1, 1))

    x_train = theano.shared(np.array(X), name="x_train")
    y_train = theano.shared(np.array(Y), name="x_train")
    print y_train.get_value(borrow=True).shape
    return x_train, y_train


n_samples=100
x_train, y_train = generate_training_data(300, n_samples=n_samples)
learner = generate_learner(x_train, y_train, param_updates)

def train(max_epochs=1000):
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        computed_costs = []
        for idx in range(n_samples):
            example_cost = learner(idx, 0.5)
            computed_costs.append(example_cost[0])
        print "epoch:%s cost:%s" % (epoch, np.mean(computed_costs))


train()