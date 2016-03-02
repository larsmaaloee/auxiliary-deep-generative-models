import cPickle as pkl
import lasagne
import numpy as np
import theano
import theano.tensor as T
from utils import env_paths as paths
from collections import OrderedDict


class Model(object):
    """
    The :class:'Model' class represents a model following the basic deep learning priciples.
    It should be subclassed when implementing new types of models.
    """

    def __init__(self, n_in, n_hidden, n_out, trans_func):
        """
        Initialisation of the basic architecture and programmatic settings of any model.
        This method should be called from any subsequent inheriting model.
        :param n_in: The input features in the model, e.g. 784.
        :param n_hidden: List containing the number of hidden units in the model, e.g. [500,500].
        :param n_out: The output units in the model, e.g. 10.
        :param batch_size: The size of the batches in the training and test sets, e.g. 100.
        :param trans_func: The transfer function for each hidden layer (cf. nonliniarities.py), e.g. sigmoid.
        """

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.transf = trans_func

        self.model_params = None

        # Model state serialisation and logging variables.
        self.model_name = self.__class__.__name__
        self.root_path = None

    def get_root_path(self):
        """
        The root path of the model is where serialization, plots etc are saved.
        :return: The root path of the model.
        """
        if self.root_path is None:
            self.root_path = paths.create_root_output_path(self.model_name, self.n_in, self.n_hidden, self.n_out)
        return self.root_path

    def build_model(self, train_set, test_set, validation_set):
        """
        Building the model should be done prior to training. It will implement the training, testing and validation
        functions.
        This method should be called from any subsequent inheriting model.
        :param loss: The loss funciton applied to training (cf. updates.py), e.g. mse.
        :param update: The update function (optimization framework) used for training (cf. updates.py), e.g. sgd.
        :param update_args: The args for the update function applied to training, e.g. (0.001,).
        """
        print "### BUILDING MODEL ###"

        self.train_args = {}
        self.train_args['inputs'] = OrderedDict({})
        self.train_args['outputs'] = OrderedDict({})

        self.test_args = {}
        self.test_args['inputs'] = OrderedDict({})
        self.test_args['outputs'] = OrderedDict({})

        self.validate_args = {}
        self.validate_args['inputs'] = OrderedDict({})
        self.validate_args['outputs'] = OrderedDict({})

        self.sym_index = T.iscalar('index')
        self.sym_batchsize = T.iscalar('batchsize')
        self.sym_lr = T.scalar('learningrate')
        self.batch_slice = slice(self.sym_index * self.sym_batchsize, (self.sym_index + 1) * self.sym_batchsize)

        self.sh_train_x = theano.shared(np.asarray(train_set[0], dtype=theano.config.floatX), borrow=True)
        if train_set[1] is not None:
            self.sh_train_t = theano.shared(np.asarray(train_set[1], dtype=theano.config.floatX), borrow=True)
        self.sh_test_x = theano.shared(np.asarray(test_set[0], dtype=theano.config.floatX), borrow=True)
        if test_set[1] is not None:
            self.sh_test_t = theano.shared(np.asarray(test_set[1], dtype=theano.config.floatX), borrow=True)
        if validation_set is not None:
            self.sh_valid_x = theano.shared(np.asarray(validation_set[0], dtype=theano.config.floatX), borrow=True)
            if validation_set[1] is not None:
                self.sh_valid_t = theano.shared(np.asarray(validation_set[1], dtype=theano.config.floatX), borrow=True)


    def dump_model(self, epoch=None):
        """
        Dump the model into a pickled version in the model path formulated in the initialisation method.
        """
        p = paths.get_model_path(self.get_root_path(), self.model_name, self.n_in, self.n_hidden, self.n_out)
        if not epoch is None: p += "_epoch_%i" % epoch
        if self.model_params is None:
            raise ("Model params are not set and can therefore not be pickled.")
        model_params = [param.get_value() for param in self.model_params]
        pkl.dump(model_params, open(p, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

    def load_model(self, id):
        """
        Load the pickled version of the model into a 'new' model instance.
        :param id: The model ID is constructed from the timestamp when the model was defined.
        """
        model_params = (self.model_name, self.n_in, self.n_hidden, self.n_out, id)
        root = paths.get_root_output_path(*model_params)
        self.root_path = root
        p = paths.get_model_path(root, *model_params[:-1])
        model_params = pkl.load(open(p, "rb"))
        for i in range(len(self.model_params)):
            init_param = self.model_params[i]
            loaded_param = model_params[i]
            if not loaded_param.shape == tuple(init_param.shape.eval()):
                print "Model could not be loaded, since parameters are not aligned."
            self.model_params[i].set_value(np.asarray(model_params[i], dtype=theano.config.floatX), borrow=True)

    def get_output(self, x):
        """
        Get output data_loaders for the model.
        :param x: The input data_loaders.
        :return: The output data_loaders.
        """
        return lasagne.layers.get_output(self.model_params, x, deterministic=True)

    def get_model_shape(self, params):
        """
        Get shape of model given the params.
        :param params: All params of the model.
        :return: List containing the shapes of the model.
        """
        w_params = [w for w in params if 'W' in str(w)]
        shapes = []
        for i in range(len(w_params)):
            shapes += [int(w_params[i].shape[0].eval())]
            if i == len(w_params) - 1:
                shapes += [int(w_params[i].shape[1].eval())]
        return shapes

    def model_info(self):
        """
        Return the model info & training specifications for the model.
        :return: The collected info string.
        """
        return ""

    def after_epoch(self):
        pass
