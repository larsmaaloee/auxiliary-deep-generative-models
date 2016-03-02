Auxiliary Deep Generatives Models
=======
This repository is the implementation of the article on `Auxiliary Deep Generative Models <http://arxiv.org/abs/1602.05473>`_.


The implementation is build on the `Parmesan <https://github.com/casperkaae/parmesan>`_, `Lasagne <http://github.com/Lasagne/Lasagne>`_ and `Theano <https://github.com/Theano/Theano>`_ libraries.


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.


**Install**


.. code-block:: bash

  git clone https://github.com/casperkaae/parmesan.git
  cd parmesan
  python setup.py develop
  pip install numpy
  pip install seaborn
  pip install matplotlib
  pip install https://github.com/Theano/Theano/archive/master.zip
  pip install https://github.com/Lasagne/Lasagne/archive/master.zip


Examples
-------------
The repository primarily includes a script running a new model on the MNIST dataset with only 100 labels - *run_sdgmssl_mnist.py*.

Please see the source code and code examples for further details. For some visualisations of the latent space and the
half moon classification examples, see https://youtu.be/g-c-xOmA2nA, https://youtu.be/hnhkKTSdmls and https://youtu.be/O8-VYr4CxsI.

