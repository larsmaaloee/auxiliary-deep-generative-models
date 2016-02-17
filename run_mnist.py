import theano.sandbox.cuda  # TODO delete
theano.sandbox.cuda.use('gpu2')  # TODO delete
import theano
from training.train import TrainModel
from lasagne_extensions.nonlinearities import rectify
from data_loaders import mnist
from models import SDGMSSL
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def run_sdgmssl_mnist():
    """
    Train an auxiliary deep generative model on the mnist dataset with 100 evenly distributed labels.
    """
    seed = np.random.randint(1, 2147462579)
    n_labeled = 100  # The total number of labeled data points.
    mnist_data = mnist.load_semi_supervised(n_labeled=n_labeled, filter_std=0.0, seed=seed, train_valid_combine=True)

    n, n_x = mnist_data[0][0].shape  # Datapoints in the dataset, input features.
    n_samples = 100  # The number of sampled labeled data points for each batch.
    n_batches = n / 100  # The number of batches.
    bs = n / n_batches  # The batchsize.

    # Initialize the auxiliary deep generative model.
    model = SDGMSSL(n_x=n_x, n_a=100, n_z=100, n_y=10, qa_hid=[500, 500],
                    qz_hid=[500, 500], qy_hid=[500, 500], px_hid=[500, 500], pa_hid=[500, 500],
                    trans_func=rectify, batchnorm=False, x_dist='bernoulli')

    # Get the training functions.
    f_train, f_test, f_validate, train_args, test_args, validate_args = model.build_model(*mnist_data)
    # Update the default function arguments.
    train_args['inputs']['batchsize_unlabeled'] = bs
    train_args['inputs']['batchsize_labeled'] = n_samples
    train_args['inputs']['beta'] = .1
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1
    test_args['inputs']['samples'] = 1
    validate_args['inputs']['samples'] = 1

    # Evaluate the approximated classification error with 100 MC samples for a good estimate
    # and output PCA plot of variable a.
    def custom_evaluation(model, path):
        mean_evals = model.get_output(mnist_data[2][0], 100)
        t_class = np.argmax(mnist_data[2][1], axis=1)
        y_class = np.argmax(mean_evals, axis=1)
        missclass = (np.sum(y_class != t_class, dtype='float32') / len(y_class)) * 100.
        train.write_to_logger("test 100-samples: %0.2f%%." % missclass)

        a = model.f_qa(mnist_data[2][0], 1)
        pca = PCA(n_components=2)
        pca_a = pca.fit_transform(a)
        plt.figure()
        color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, 10)))
        for i in range(10):
            c = next(color)
            x = pca_a[np.argmax(mnist_data[2][1], axis=1) == i, :]
            plt.scatter(x[:, 0], x[:, 1], c=c, s=5., lw=0, marker='o', label=str(i))
        plt.legend()
        plt.savefig(path.replace(".png", "") + "_a_dist.png", format='png')

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=10, f_custom_eval=custom_evaluation)
    train.add_initial_training_notes("Training the auxiliary deep generative model with %i labels. bn %s. seed %i." % (
    n_labeled, str(model.batchnorm), seed))
    train.train_model(f_train, train_args,
                      f_test, test_args,
                      f_validate, validate_args,
                      n_train_batches=n_batches,
                      n_epochs=1000,
                      # Any symbolic model variable can be annealed during
                      # training with a tuple of (var_name, every, scale constant, minimum value).
                      anneal=[("learningrate", 200, 0.75, 3e-5)])

if __name__ == "__main__":
    run_sdgmssl_mnist()
