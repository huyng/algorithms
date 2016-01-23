"""
A vanilla RNN

I will now add mini batches
"""

import sys
import numpy as np
import theano.tensor as T
import theano
import datasets
from numpy.random import uniform
from theano import config

# p = sys.stderr.write
# sys.stdout = open(os.devnull, "w")

# ======
# Config
# ======

# theano.config.mode='FAST_COMPILE'
# theano.config.exception_verbosity="high"
# theano.config.compute_test_value='off'
theano.config.floatX = 'float32'


# ============
# Create model
# ============
n_x = 1      # input activations size for a single time step
n_h = 64     # hidden activations size for a single time step
n_y = 1      # output activation size for a single time step

# http://www.schraudolph.org/pubs/GerSchSch02.pdf (mentions 10 time steps already a big problem for Simple RNNs)
n_steps = 8 # number of timesteps


# inputs/outputs
# ==============
x = T.tensor3(name='x', dtype=config.floatX)  # (n_steps, n_x, n_samples)
y = T.tensor3(name='y', dtype=config.floatX)  # (n_steps, n_x, n_samples)


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

[h, y_predict], _ = theano.scan(step, sequences=x, outputs_info=[T.alloc(h0, 32, n_h), None], n_steps=n_steps)

lr = T.scalar('lr', dtype=theano.config.floatX)
loss =  T.mean((y_predict - y) ** 2, axis=0).sum()
L1 = abs(W_xh.sum()) + abs(W_hh.sum())  + abs(W_hy.sum())
L2 = (W_xh**2).sum() + (W_hh**2).sum() + (W_hy**2).sum()
L1_reg = 0.0
L2_reg = 0.0
cost = loss



# symbolically generate the derivative of the cost
# w.r.t the parameters (AKA the gradient) and the 
# updates for each parameter
param_updates = []
for param in parameters:
    gradient = T.grad(cost, param)
    # gradient = theano.printing.Print("d(%s)" % param.name)(gradient)
    update = param - lr*gradient
    param_updates.append((param, update))


# create a function that performs a single gradient descent step on training data
def make_learner(x_train, y_train, param_updates):
    index = T.lscalar('index')
    batch_size = 32
    learner_fn = theano.function(
            inputs=[index, lr],
            outputs=[cost, y_predict],
            updates=param_updates,
            givens={
                x: x_train[:, index*batch_size:(index+1)*batch_size, : ],
                y: y_train[:, index*batch_size:(index+1)*batch_size, : ]
            }
        )

    # test_fn = theano.function(
    #     inputs=[x, y],
    #     outputs=[cost],
    # )
    # return learner_fn, test_fn
    return learner_fn, learner_fn

def make_predictor():
    predict_fn = theano.function(
            inputs=[x],
            outputs=[y_predict],
        )
    return predict_fn


def train(x_train_val, y_train_val, x_test_val, y_test_val):

    # data comes in as (n_samples, n_steps, n_x)
    # reshape data so that it's shaped as (n_steps, n_samples, n_x)
    x_train_val = x_train_val.transpose(1,0,2)
    y_train_val = y_train_val.transpose(1,0,2)

    x_train = theano.shared(x_train_val)
    y_train = theano.shared(y_train_val)
    n_train_samples = x_train_val.shape[-1]
    # n_test_samples = x_test_val.shape[-1]

    # creating training and test function
    learn_fn, test_fn = make_learner(x_train, y_train, param_updates)

    # start gradient descent w/ batch_size==1
    max_epochs = 40000
    lr_val = 0.1 # http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf

    for epoch in range(max_epochs):
        train_costs = []
        for idx in range(n_train_samples):
            mini_batch_cost, y_predicted_val = learn_fn(idx, lr_val)
            print y_predicted_val.shape
            print y_predicted_val

            train_costs.append(mini_batch_cost)

        # test_costs = []
        # for idx in range(n_test_samples):
        #     sample_cost = test_fn(x_test_val[idx], y_test_val[idx])
        #     test_costs.append(sample_cost)

        report({
            '_epoch':epoch,
            'trn': float(np.mean(train_costs)),
            'tst': float(0)
            # 'tst': float(np.mean(test_costs))
        })

        # checkpoint parameters every 1000 epochs
        if epoch % 1000 == 0:
            save_parameters(epoch)



# ==============
# Infrastructure
# ==============
def save_parameters(epoch=None):
    d = {}
    fname = "weights.npz" if epoch is None else "weights.%d.npz" % epoch
    for p in parameters:
        k = p.name
        d[k] = p.get_value(borrow=False)
    np.savez(fname, **d)

def load_parameters(epoch=None):
    fname = "weights.npz" if epoch is None else "weights.%d.npz" % epoch
    f = np.load(fname)
    for p in parameters:
        p.set_value(f[p.name])

def report(d):
    import json
    with open("learning_curve.txt", "a") as fh:
        fh.write(json.dumps(d))
        fh.write("\n")
    print "epoch=%-6s -- trn_cost=%0.8f tst_cost=%0.8f" % (d['_epoch'], d['trn'], d['tst'])


if __name__ == '__main__':
    # generate some simple training data
    import uuid
    run_name = uuid.uuid4().hex[:8]
    n_samples = 10000
    x_train_val, y_train_val = datasets.sinewaves(n_steps, n=n_samples)
    x_test_val, y_test_val = datasets.sinewaves(n_steps, n=20)          # (n_samples, n_steps, n_x)
    train(x_train_val, y_train_val, x_test_val, y_test_val)


