import theano
import theano.tensor as T
from lasagne import init
from base import Model
from lasagne_extensions.layers import (SampleLayer, MultinomialLogDensityLayer,
                                       GaussianLogDensityLayer, StandardNormalLogDensityLayer, BernoulliLogDensityLayer,
                                       InputLayer, DenseLayer, DimshuffleLayer, ElemwiseSumLayer, ReshapeLayer,
                                       NonlinearityLayer, BatchNormLayer, get_all_params, get_output)
from lasagne_extensions.objectives import categorical_crossentropy, categorical_accuracy
from lasagne_extensions.nonlinearities import rectify, softplus, sigmoid, softmax
from lasagne_extensions.updates import total_norm_constraint
from lasagne_extensions.updates import adam
from parmesan.distributions import log_normal
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


class SDGMSSL(Model):
    """
    The :class:'SDGMSSL' class represents the implementation of the model described in the
    Auxiliary Generative Models article on Arxiv.org.
    """

    def __init__(self, n_x, n_a, n_z, n_y, qa_hid, qz_hid, qy_hid, px_hid, pa_hid, nonlinearity=rectify,
                 px_nonlinearity=None, x_dist='bernoulli', batchnorm=False, seed=1234):
        """
        Initialize an skip deep generative model consisting of
        discriminative classifier q(y|a,x),
        generative model P p(a|z,y) and p(x|a,z,y),
        inference model Q q(a|x) and q(z|a,x,y).
        Weights are initialized using the Bengio and Glorot (2010) initialization scheme.
        :param n_x: Number of inputs.
        :param n_a: Number of auxiliary.
        :param n_z: Number of latent.
        :param n_y: Number of classes.
        :param qa_hid: List of number of deterministic hidden q(a|x).
        :param qz_hid: List of number of deterministic hidden q(z|a,x,y).
        :param qy_hid: List of number of deterministic hidden q(y|a,x).
        :param px_hid: List of number of deterministic hidden p(a|z,y) & p(x|z,y).
        :param nonlinearity: The transfer function used in the deterministic layers.
        :param x_dist: The x distribution, 'bernoulli', 'multinomial', or 'gaussian'.
        :param batchnorm: Boolean value for batch normalization.
        :param seed: The random seed.
        """
        super(SDGMSSL, self).__init__(n_x, qz_hid + px_hid, n_a + n_z, nonlinearity)
        self.x_dist = x_dist
        self.n_y = n_y
        self.n_x = n_x
        self.n_a = n_a
        self.n_z = n_z
        self.batchnorm = batchnorm
        self._srng = RandomStreams(seed)

        # Decide Glorot initializaiton of weights.
        init_w = 1e-3
        hid_w = ""
        if nonlinearity == rectify or nonlinearity == softplus:
            hid_w = "relu"

        # Define symbolic variables for theano functions.
        self.sym_beta = T.scalar('beta')  # scaling constant beta
        self.sym_x_l = T.matrix('x')  # labeled inputs
        self.sym_t_l = T.matrix('t')  # labeled targets
        self.sym_x_u = T.matrix('x')  # unlabeled inputs
        self.sym_bs_l = T.iscalar('bs_l')  # number of labeled data
        self.sym_samples = T.iscalar('samples')  # MC samples
        self.sym_z = T.matrix('z')  # latent variable z
        self.sym_a = T.matrix('a')  # auxiliary variable a

        # Assist methods for collecting the layers
        def dense_layer(layer_in, n, dist_w=init.GlorotNormal, dist_b=init.Normal):
            dense = DenseLayer(layer_in, n, dist_w(hid_w), dist_b(init_w), None)
            if batchnorm:
                dense = BatchNormLayer(dense)
            return NonlinearityLayer(dense, self.transf)

        def stochastic_layer(layer_in, n, samples, nonlin=None):
            mu = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            logvar = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            return SampleLayer(mu, logvar, eq_samples=samples, iw_samples=1), mu, logvar

        # Input layers
        l_x_in = InputLayer((None, n_x))
        l_y_in = InputLayer((None, n_y))

        # Auxiliary q(a|x)
        l_qa_x = l_x_in
        for hid in qa_hid:
            l_qa_x = dense_layer(l_qa_x, hid)
        l_qa_x, l_qa_x_mu, l_qa_x_logvar = stochastic_layer(l_qa_x, n_a, self.sym_samples)

        # Classifier q(y|a,x)
        l_qa_to_qy = DenseLayer(l_qa_x, qy_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_qa_to_qy = ReshapeLayer(l_qa_to_qy, (-1, self.sym_samples, 1, qy_hid[0]))
        l_x_to_qy = DenseLayer(l_x_in, qy_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_x_to_qy = DimshuffleLayer(l_x_to_qy, (0, 'x', 'x', 1))
        l_qy_xa = ReshapeLayer(ElemwiseSumLayer([l_qa_to_qy, l_x_to_qy]), (-1, qy_hid[0]))
        if batchnorm:
            l_qy_xa = BatchNormLayer(l_qy_xa)
        l_qy_xa = NonlinearityLayer(l_qy_xa, self.transf)
        if len(qy_hid) > 1:
            for hid in qy_hid[1:]:
                l_qy_xa = dense_layer(l_qy_xa, hid)
        l_qy_xa = DenseLayer(l_qy_xa, n_y, init.GlorotNormal(), init.Normal(init_w), softmax)

        # Recognition q(z|x,a,y)
        l_qa_to_qz = DenseLayer(l_qa_x, qz_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_qa_to_qz = ReshapeLayer(l_qa_to_qz, (-1, self.sym_samples, 1, qz_hid[0]))
        l_x_to_qz = DenseLayer(l_x_in, qz_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_x_to_qz = DimshuffleLayer(l_x_to_qz, (0, 'x', 'x', 1))
        l_y_to_qz = DenseLayer(l_y_in, qz_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_y_to_qz = DimshuffleLayer(l_y_to_qz, (0, 'x', 'x', 1))
        l_qz_axy = ReshapeLayer(ElemwiseSumLayer([l_qa_to_qz, l_x_to_qz, l_y_to_qz]), (-1, qz_hid[0]))
        if batchnorm:
            l_qz_axy = BatchNormLayer(l_qz_axy)
        l_qz_axy = NonlinearityLayer(l_qz_axy, self.transf)
        if len(qz_hid) > 1:
            for hid in qz_hid[1:]:
                l_qz_axy = dense_layer(l_qz_axy, hid)
        l_qz_axy, l_qz_axy_mu, l_qz_axy_logvar = stochastic_layer(l_qz_axy, n_z, 1)

        # Generative p(a|z,y)
        l_y_to_pa = DenseLayer(l_y_in, pa_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_y_to_pa = DimshuffleLayer(l_y_to_pa, (0, 'x', 'x', 1))
        l_qz_to_pa = DenseLayer(l_qz_axy, pa_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_qz_to_pa = ReshapeLayer(l_qz_to_pa, (-1, self.sym_samples, 1, pa_hid[0]))
        l_pa_zy = ReshapeLayer(ElemwiseSumLayer([l_qz_to_pa, l_y_to_pa]), [-1, pa_hid[0]])
        if batchnorm:
            l_pa_zy = BatchNormLayer(l_pa_zy)
        l_pa_zy = NonlinearityLayer(l_pa_zy, self.transf)
        if len(pa_hid) > 1:
            for hid in pa_hid[1:]:
                l_pa_zy = dense_layer(l_pa_zy, hid)
        l_pa_zy, l_pa_zy_mu, l_pa_zy_logvar = stochastic_layer(l_pa_zy, n_a, 1)

        # Generative p(x|a,z,y)
        l_qa_to_px = DenseLayer(l_qa_x, px_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_qa_to_px = ReshapeLayer(l_qa_to_px, (-1, self.sym_samples, 1, px_hid[0]))
        l_y_to_px = DenseLayer(l_y_in, px_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_y_to_px = DimshuffleLayer(l_y_to_px, (0, 'x', 'x', 1))
        l_qz_to_px = DenseLayer(l_qz_axy, px_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_qz_to_px = ReshapeLayer(l_qz_to_px, (-1, self.sym_samples, 1, px_hid[0]))
        l_px_azy = ReshapeLayer(ElemwiseSumLayer([l_qa_to_px, l_qz_to_px, l_y_to_px]), [-1, px_hid[0]])
        if batchnorm:
            l_px_azy = BatchNormLayer(l_px_azy)
        l_px_azy = NonlinearityLayer(l_px_azy, self.transf)
        if len(px_hid) > 1:
            for hid in px_hid[1:]:
                l_px_azy = dense_layer(l_px_azy, hid)

        if x_dist == 'bernoulli':
            l_px_azy = DenseLayer(l_px_azy, n_x, init.GlorotNormal(), init.Normal(init_w), sigmoid)
        elif x_dist == 'multinomial':
            l_px_azy = DenseLayer(l_px_azy, n_x, init.GlorotNormal(), init.Normal(init_w), softmax)
        elif x_dist == 'gaussian':
            l_px_azy, l_px_zy_mu, l_px_zy_logvar = stochastic_layer(l_px_azy, n_x, 1, px_nonlinearity)

        # Reshape all the model layers to have the same size
        self.l_x_in = l_x_in
        self.l_y_in = l_y_in
        self.l_a_in = l_qa_x

        self.l_qa = ReshapeLayer(l_qa_x, (-1, self.sym_samples, 1, n_a))
        self.l_qa_mu = DimshuffleLayer(l_qa_x_mu, (0, 'x', 'x', 1))
        self.l_qa_logvar = DimshuffleLayer(l_qa_x_logvar, (0, 'x', 'x', 1))

        self.l_qz = ReshapeLayer(l_qz_axy, (-1, self.sym_samples, 1, n_z))
        self.l_qz_mu = ReshapeLayer(l_qz_axy_mu, (-1, self.sym_samples, 1, n_z))
        self.l_qz_logvar = ReshapeLayer(l_qz_axy_logvar, (-1, self.sym_samples, 1, n_z))

        self.l_qy = ReshapeLayer(l_qy_xa, (-1, self.sym_samples, 1, n_y))

        self.l_pa = ReshapeLayer(l_pa_zy, (-1, self.sym_samples, 1, n_a))
        self.l_pa_mu = ReshapeLayer(l_pa_zy_mu, (-1, self.sym_samples, 1, n_a))
        self.l_pa_logvar = ReshapeLayer(l_pa_zy_logvar, (-1, self.sym_samples, 1, n_a))

        self.l_px = ReshapeLayer(l_px_azy, (-1, self.sym_samples, 1, n_x))
        self.l_px_mu = ReshapeLayer(l_px_zy_mu, (-1, self.sym_samples, 1, n_x)) if x_dist == "gaussian" else None
        self.l_px_logvar = ReshapeLayer(l_px_zy_logvar,
                                        (-1, self.sym_samples, 1, n_x)) if x_dist == "gaussian" else None

        # Predefined functions
        inputs = [self.sym_x_l, self.sym_samples]
        outputs = get_output(self.l_qy, self.sym_x_l, deterministic=True).mean(axis=(1, 2))
        self.f_qy = theano.function(inputs, outputs)

        inputs = [self.sym_x_l, self.sym_samples]
        outputs = get_output(self.l_qa, self.sym_x_l, deterministic=True).mean(axis=(1, 2))
        self.f_qa = theano.function(inputs, outputs)

        inputs = {l_qz_axy: self.sym_z, l_y_in: self.sym_t_l}
        outputs = get_output(self.l_pa, inputs, deterministic=True)
        self.f_pa = theano.function([self.sym_z, self.sym_t_l, self.sym_samples], outputs)

        inputs = {l_qa_x: self.sym_a, l_qz_axy: self.sym_z, l_y_in: self.sym_t_l}
        outputs = get_output(self.l_px, inputs, deterministic=True)
        self.f_px = theano.function([self.sym_a, self.sym_z, self.sym_t_l, self.sym_samples], outputs)

        # Define model parameters
        self.model_params = get_all_params([self.l_qy, self.l_pa, self.l_px])
        self.trainable_model_params = get_all_params([self.l_qy, self.l_pa, self.l_px], trainable=True)

    def build_model(self, train_set_unlabeled, train_set_labeled, test_set, validation_set=None):
        """
        Build the auxiliary deep generative model from the initialized hyperparameters.
        Define the lower bound term and compile it into a training function.
        :param train_set_unlabeled: Unlabeled train set containing variables x, t.
        :param train_set_labeled: Unlabeled train set containing variables x, t.
        :param test_set: Test set containing variables x, t.
        :param validation_set: Validation set containing variables x, t.
        :return: train, test, validation function and dicts of arguments.
        """
        super(SDGMSSL, self).build_model(train_set_unlabeled, test_set, validation_set)

        sh_train_x_l = theano.shared(np.asarray(train_set_labeled[0], dtype=theano.config.floatX), borrow=True)
        sh_train_t_l = theano.shared(np.asarray(train_set_labeled[1], dtype=theano.config.floatX), borrow=True)
        n = self.sh_train_x.shape[0].astype(theano.config.floatX)  # no. of data points
        n_l = sh_train_x_l.shape[0].astype(theano.config.floatX)  # no. of labeled data points

        # Define the layers for the density estimation used in the lower bound.
        l_log_qa = GaussianLogDensityLayer(self.l_qa, self.l_qa_mu, self.l_qa_logvar)
        l_log_qz = GaussianLogDensityLayer(self.l_qz, self.l_qz_mu, self.l_qz_logvar)
        l_log_qy = MultinomialLogDensityLayer(self.l_qy, self.l_y_in, eps=1e-8)

        l_log_pz = StandardNormalLogDensityLayer(self.l_qz)
        l_log_pa = GaussianLogDensityLayer(self.l_qa, self.l_pa_mu, self.l_pa_logvar)
        if self.x_dist == 'bernoulli':
            l_log_px = BernoulliLogDensityLayer(self.l_px, self.l_x_in)
        elif self.x_dist == 'multinomial':
            l_log_px = MultinomialLogDensityLayer(self.l_px, self.l_x_in)
        elif self.x_dist == 'gaussian':
            l_log_px = GaussianLogDensityLayer(self.l_x_in, self.l_px_mu, self.l_px_logvar)

        def lower_bound(log_pa, log_qa, log_pz, log_qz, log_py, log_px):
            lb = log_px + log_py + log_pz + log_pa - log_qa - log_qz
            return lb

        # Lower bound for labeled data
        out_layers = [l_log_pa, l_log_pz, l_log_qa, l_log_qz, l_log_px, l_log_qy]
        inputs = {self.l_x_in: self.sym_x_l, self.l_y_in: self.sym_t_l}
        out = get_output(out_layers, inputs, batch_norm_update_averages=False, batch_norm_use_averages=False)
        log_pa_l, log_pz_l, log_qa_x_l, log_qz_axy_l, log_px_zy_l, log_qy_ax_l = out
        # Prior p(y) expecting that all classes are evenly distributed
        py_l = softmax(T.zeros((self.sym_x_l.shape[0], self.n_y)))
        log_py_l = -categorical_crossentropy(py_l, self.sym_t_l).reshape((-1, 1)).dimshuffle((0, 'x', 'x', 1))
        lb_l = lower_bound(log_pa_l, log_qa_x_l, log_pz_l, log_qz_axy_l, log_py_l, log_px_zy_l)
        lb_l = lb_l.mean(axis=(1, 2))  # Mean over the sampling dimensions
        log_qy_ax_l *= (self.sym_beta * (n / n_l))  # Scale the supervised cross entropy with the alpha constant
        lb_l -= log_qy_ax_l.mean(axis=(1, 2))  # Collect the lower bound term and mean over sampling dimensions

        # Lower bound for unlabeled data
        bs_u = self.sym_x_u.shape[0]

        # For the integrating out approach, we repeat the input matrix x, and construct a target (bs * n_y) x n_y
        # Example of input and target matrix for a 3 class problem and batch_size=2. 2D tensors of the form
        #               x_repeat                     t_repeat
        #  [[x[0,0], x[0,1], ..., x[0,n_x]]         [[1, 0, 0]
        #   [x[1,0], x[1,1], ..., x[1,n_x]]          [1, 0, 0]
        #   [x[0,0], x[0,1], ..., x[0,n_x]]          [0, 1, 0]
        #   [x[1,0], x[1,1], ..., x[1,n_x]]          [0, 1, 0]
        #   [x[0,0], x[0,1], ..., x[0,n_x]]          [0, 0, 1]
        #   [x[1,0], x[1,1], ..., x[1,n_x]]]         [0, 0, 1]]
        t_eye = T.eye(self.n_y, k=0)
        t_u = t_eye.reshape((self.n_y, 1, self.n_y)).repeat(bs_u, axis=1).reshape((-1, self.n_y))
        x_u = self.sym_x_u.reshape((1, bs_u, self.n_x)).repeat(self.n_y, axis=0).reshape((-1, self.n_x))

        # Since the expectation of var a is outside the integration we calculate E_q(a|x) first
        a_x_u = get_output(self.l_qa, self.sym_x_u, batch_norm_update_averages=True, batch_norm_use_averages=False)
        a_x_u_rep = a_x_u.reshape((1, bs_u * self.sym_samples, self.n_a)).repeat(self.n_y, axis=0).reshape(
            (-1, self.n_a))
        out_layers = [l_log_pa, l_log_pz, l_log_qa, l_log_qz, l_log_px]
        inputs = {self.l_x_in: x_u, self.l_y_in: t_u, self.l_a_in: a_x_u_rep}
        out = get_output(out_layers, inputs, batch_norm_update_averages=False, batch_norm_use_averages=False)
        log_pa_u, log_pz_u, log_qa_x_u, log_qz_axy_u, log_px_zy_u = out
        # Prior p(y) expecting that all classes are evenly distributed
        py_u = softmax(T.zeros((bs_u * self.n_y, self.n_y)))
        log_py_u = -categorical_crossentropy(py_u, t_u).reshape((-1, 1)).dimshuffle((0, 'x', 'x', 1))
        lb_u = lower_bound(log_pa_u, log_qa_x_u, log_pz_u, log_qz_axy_u, log_py_u, log_px_zy_u)
        lb_u = lb_u.reshape((self.n_y, 1, 1, bs_u)).transpose(3, 1, 2, 0).mean(axis=(1, 2))
        inputs = {self.l_x_in: self.sym_x_u, self.l_a_in: a_x_u.reshape((-1, self.n_a))}
        y_u = get_output(self.l_qy, inputs, batch_norm_update_averages=True, batch_norm_use_averages=False).mean(
            axis=(1, 2))
        y_u += 1e-8  # Ensure that we get no NANs when calculating the entropy
        y_u /= T.sum(y_u, axis=1, keepdims=True)
        lb_u = (y_u * (lb_u - T.log(y_u))).sum(axis=1)

        if self.batchnorm:
            # TODO: implement the BN layer correctly.
            inputs = {self.l_x_in: self.sym_x_u, self.l_y_in: y_u, self.l_a_in: a_x_u}
            get_output(out_layers, inputs, weighting=None, batch_norm_update_averages=True,
                       batch_norm_use_averages=False)

        # Regularizing with weight priors p(theta|N(0,1)), collecting and clipping gradients
        weight_priors = 0.0
        for p in self.trainable_model_params:
            if 'W' not in str(p):
                continue
            weight_priors += log_normal(p, 0, 1).sum()

        # Collect the lower bound and scale it with the weight priors.
        elbo = ((lb_l.mean() + lb_u.mean()) * n + weight_priors) / -n
        lb_labeled = -lb_l.mean()
        lb_unlabeled = -lb_u.mean()

        grads_collect = T.grad(elbo, self.trainable_model_params)
        params_collect = self.trainable_model_params
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads_collect, max_norm=max_norm)
        mgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        updates = adam(mgrads, params_collect, self.sym_lr, sym_beta1, sym_beta2)

        # Training function
        indices = self._srng.choice(size=[self.sym_bs_l], a=sh_train_x_l.shape[0], replace=False)
        x_batch_l = sh_train_x_l[indices]
        t_batch_l = sh_train_t_l[indices]
        x_batch_u = self.sh_train_x[self.batch_slice]
        if self.x_dist == 'bernoulli':  # Sample bernoulli input.
            x_batch_u = self._srng.binomial(size=x_batch_u.shape, n=1, p=x_batch_u, dtype=theano.config.floatX)
            x_batch_l = self._srng.binomial(size=x_batch_l.shape, n=1, p=x_batch_l, dtype=theano.config.floatX)

        givens = {self.sym_x_l: x_batch_l,
                  self.sym_x_u: x_batch_u,
                  self.sym_t_l: t_batch_l}
        inputs = [self.sym_index, self.sym_batchsize, self.sym_bs_l, self.sym_beta,
                  self.sym_lr, sym_beta1, sym_beta2, self.sym_samples]
        outputs = [elbo, lb_labeled, lb_unlabeled]
        f_train = theano.function(inputs=inputs, outputs=outputs, givens=givens, updates=updates)

        # Default training args. Note that these can be changed during or prior to training.
        self.train_args['inputs']['batchsize_unlabeled'] = 100
        self.train_args['inputs']['batchsize_labeled'] = 100
        self.train_args['inputs']['beta'] = 0.1
        self.train_args['inputs']['learningrate'] = 3e-4
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['inputs']['samples'] = 1
        self.train_args['outputs']['lb'] = '%0.4f'
        self.train_args['outputs']['lb-labeled'] = '%0.4f'
        self.train_args['outputs']['lb-unlabeled'] = '%0.4f'

        # Validation and test function
        y = get_output(self.l_qy, self.sym_x_l, deterministic=True).mean(axis=(1, 2))
        class_err = (1. - categorical_accuracy(y, self.sym_t_l).mean()) * 100
        givens = {self.sym_x_l: self.sh_test_x,
                  self.sym_t_l: self.sh_test_t}
        f_test = theano.function(inputs=[self.sym_samples], outputs=[class_err], givens=givens)

        # Test args.  Note that these can be changed during or prior to training.
        self.test_args['inputs']['samples'] = 1
        self.test_args['outputs']['test'] = '%0.2f%%'

        f_validate = None
        if validation_set is not None:
            givens = {self.sym_x_l: self.sh_valid_x,
                      self.sym_t_l: self.sh_valid_t}
            f_validate = theano.function(inputs=[self.sym_samples], outputs=[class_err], givens=givens)
        # Default validation args. Note that these can be changed during or prior to training.
        self.validate_args['inputs']['samples'] = 1
        self.validate_args['outputs']['validation'] = '%0.2f%%'

        return f_train, f_test, f_validate, self.train_args, self.test_args, self.validate_args

    def get_output(self, x, samples=1):
        return self.f_qy(x, samples)

    def model_info(self):
        qa_shapes = self.get_model_shape(get_all_params(self.l_qa))
        qy_shapes = self.get_model_shape(get_all_params(self.l_qy))[len(qa_shapes) - 1:]
        qz_shapes = self.get_model_shape(get_all_params(self.l_qz))[len(qa_shapes) - 1:]
        px_shapes = self.get_model_shape(get_all_params(self.l_px))[(len(qz_shapes) - 1) + (len(qa_shapes) - 1):]
        pa_shapes = self.get_model_shape(get_all_params(self.l_pa))[(len(qz_shapes) - 1) + (len(qa_shapes) - 1):]
        s = ""
        s += 'batch norm: %s.\n' % (str(self.batchnorm))
        s += 'x distribution: %s.\n' % (str(self.x_dist))
        s += 'model q(a|x): %s.\n' % str(qa_shapes)[1:-1]
        s += 'model q(z|a,x,y): %s.\n' % str(qz_shapes)[1:-1]
        s += 'model q(y|a,x): %s.\n' % str(qy_shapes)[1:-1]
        s += 'model p(x|a,z,y): %s.\n' % str(px_shapes)[1:-1]
        s += 'model p(a|z,y): %s.' % str(pa_shapes)[1:-1]
        return s
