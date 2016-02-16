from lasagne.nonlinearities import *
import theano.tensor as T

def softplus(x): return T.log(T.exp(x) + 1)