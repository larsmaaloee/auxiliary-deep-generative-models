Auxiliary Deep Generatives Models
=======
This repository is the implementation of the skip deep generative model.


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
The repository primarily includes


* script running a new model on the MNIST dataset with only 100 labels - *run_sdgmssl_mnist.py*.
* script running a new model on the SVHN dataset with only 1000 labels - *run_sdgmssl_svhn.py*.
* script running a new model on the NORB datasets with only 1000 labels - *run_sdgmssl_norb.py*.


Please see the source code and code examples for further details.