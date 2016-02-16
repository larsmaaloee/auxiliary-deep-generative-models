from os.path import join, exists, abspath
from os import mkdir, listdir
import time
import datetime
import uuid

# Root paths
def get_project_name():
    return 'auxiliary-deep-generative-models'


def get_data_path():
    project_name = get_project_name()
    full_path = abspath('.')
    path = join(full_path[:full_path.find(project_name) + len(project_name)], 'data_preparation')
    return path_exists(path)


def get_output_path():
    project_name = get_project_name()
    full_path = abspath('.')
    path = join(full_path[:full_path.find(project_name) + len(project_name)], 'output')
    return path_exists(path)


# Plotting
def get_training_evaluation_path(root_path):
    return path_exists(join(root_path, 'training evaluations'))


def get_plot_evaluation_path_for_model(root_path, extension):
    return join(get_training_evaluation_path(root_path), 'evaluation%s' % (extension))


def get_custom_eval_path(i, root_path):
    r_path = path_exists(join(root_path, 'training custom evals'))
    return join(r_path, 'custom_eval_plot_%s.png' % str(i))

def get_plot_evaluation_path():
    return join(get_output_path(), 'evaluation.png')

# Pickle
def get_pickle_path(root_path):
    return path_exists(join(root_path, 'pickled model'))


def get_model_path(root_path, type, n_in, n_hidden, n_out):
    return join(get_pickle_path(root_path), '%s_%s_%s_%s.pkl' % (type, str(n_in), str(n_hidden), str(n_out)))


# Logging
def get_logging_path(root_path):
    t = time.time()
    n = "_logging_%s.log" % datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d-%H%M%S')
    return join(root_path, n)

def find_logging_path(id):
    out = get_output_path()
    dirs = listdir(out)
    path = ''
    for d in dirs:
        if str(id) in d:
            path = join(out, d)
    if path == '':
        raise 'The ID couldn\'t be found'
    for f in listdir(path):
        if 'log' in f:
            return join(path, f)

# Root path
def get_root_output_path(type, n_in, n_hidden, n_out, id):
    root = 'id_%s_%s_%s_%s_%s' % (str(id), type, str(n_in), str(n_hidden), str(n_out))
    path = join(get_output_path(), root)
    return path


def create_root_output_path(type, n_in, n_hidden, n_out):
    t = time.time()
    d = datetime.datetime.fromtimestamp(t).strftime('%Y%m%d%H%M%S')
    root = 'id_%s_%s_%s_%s_%s' % (str(d), type, str(n_in), str(n_hidden), str(n_out))
    path = join(get_output_path(), root)
    if exists(path): path += "_(%s)" % str(uuid.uuid4())
    return path_exists(path)


def path_exists(path):
    if not exists(path):
        mkdir(path)
    return path
