# Copyright 2020 The Google Authors. All Rights Reserved.
#
# Licensed under the MIT License (the "License");
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Script to run the experiments and plot the results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sklearn.datasets
from sklearn.model_selection import train_test_split

from agbt import AGBT
from agbt_b import AGBTB
import functional as F
from gbt import GBT
from tree import Dataset
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('data_folder', None, 'The directory of datasets.')

flags.DEFINE_enum('dataset_name', 'all_datasets', [
    'all_datasets', 'a1a', 'w1a', 'housing', 'w8a', 'a9a', 'colon', 'Year',
    'rcv1'
], ('The name of instances.'
    '`all_datasets` means all of the instances in the folder.'))

flags.DEFINE_enum('loss', 'L2Loss', ['L2Loss', 'LogisticLoss'],
                  'The loss function.')

flags.DEFINE_integer(
    'early_stopping_rounds', 100000,
    ('Stop the algorithm if the validation loss does not improve after this'
     'number of iterations.'))

flags.DEFINE_float(
    'z_shrinkage_parameter', 0.1,
    'The shrinkage parameter in the z-update in accelerated method.')

flags.DEFINE_string('output_dir', None,
                    'The directory where output will be written.')

flags.DEFINE_integer('max_depth', 4, 'Maximal depth of a tree.')
flags.DEFINE_integer('num_trees', 20, 'Number of boosting iterations.')
flags.DEFINE_float('min_split_gain', 0.01, 'Minimal gain for splitting a leaf.')
flags.DEFINE_float('learning_rate', 0.3, 'Learning rate.')
flags.DEFINE_float('regularizer_const', 1, 'Regularizer constant.')
flags.DEFINE_boolean('use_hessian', False, 'Whether to use Hessian.')

TEST_SIZE = 0.2
RANDOM_STATE = 1

LOSS = {'L2Loss': F.L2Loss, 'LogisticLoss': F.LogisticLoss}


def set_up_data(data_folder, dataset_name):
  path = os.path.join(data_folder, dataset_name + '.txt')
  data = sklearn.datasets.load_svmlight_file(gfile.Open(path, mode='rb'))
  x = np.asarray(data[0].todense())
  y = np.array(data[1])
  return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def save_output(output_dict, name, params):
  dir = os.path.join(FLAGS.output_dir, 'output')
  if not gfile.Exists(dir):
    gfile.MakeDirs(dir)
  matfile_path = dir + '/{:s}_lr_{:s}_min_split_gain_{:s}_num_trees_{:s}_max_depth_{:s}.mat'.format(
      name,
      str(params.learning_rate).replace('.', ''),
      str(params.min_split_gain).replace('.', ''),
      str(params.num_trees).replace('.', ''),
      str(params.max_depth).replace('.', ''),
  )
  scipy.io.savemat(gfile.Open(matfile_path, mode='wb'), mdict=output_dict)
  return 0


def plot_figures(output_dict, name, params):
  """Plots the figure from the output."""
  figure_dir = os.path.join(FLAGS.output_dir, 'figures')
  if not gfile.Exists(figure_dir):
    gfile.MakeDirs(figure_dir)
  fig = plt.figure()
  plt.plot(output_dict['gbt_train_losses'], label='gbt')
  plt.plot(output_dict['agbt_b_train_losses'], label='agbt_b')
  plt.plot(output_dict['agbt_train_losses_1'], label='agbt1')
  plt.plot(output_dict['agbt_train_losses_2'], label='agbt01')
  plt.plot(output_dict['agbt_train_losses_3'], label='agbt001')
  plt.legend()
  fig.savefig(
      gfile.Open(
          figure_dir +
          '/train_{:s}_lr_{:s}_min_split_gain_{:s}_num_trees_{:s}'.format(
              name,
              str(params.learning_rate).replace('.', ''),
              str(params.min_split_gain).replace('.', ''),
              str(params.num_trees).replace('.', ''),
          ), 'wb'))

  fig = plt.figure()
  plt.plot(output_dict['gbt_test_losses'], label='gbt')
  plt.plot(output_dict['agbt_b_test_losses'], label='agbt_b')
  plt.plot(output_dict['agbt_train_losses_1'], label='agbt1')
  plt.plot(output_dict['agbt_train_losses_2'], label='agbt01')
  plt.plot(output_dict['agbt_train_losses_3'], label='agbt001')
  plt.legend()
  fig.savefig(
      gfile.Open(
          figure_dir +
          'test_{:s}_lr_{:s}_min_split_gain_{:s}_num_trees_{:s}'.format(
              name,
              str(params.learning_rate).replace('.', ''),
              str(params.min_split_gain).replace('.', ''),
              str(params.num_trees).replace('.', ''),
          ), 'wb'))
  fig = plt.figure()
  plt.plot(output_dict['gbt_train_losses'], label='gbt')
  plt.plot(output_dict['agbt_b_train_losses'], label='agbt_b')
  plt.plot(output_dict['agbt_train_losses_1'], label='agbt1')
  plt.plot(output_dict['agbt_train_losses_2'], label='agbt01')
  plt.plot(output_dict['agbt_train_losses_3'], label='agbt001')
  plt.yscale('log')
  plt.legend()
  fig.savefig(
      gfile.Open(
          figure_dir +
          'log_train_{:s}_lr_{:s}_min_split_gain_{:s}_num_trees_{:s}'.format(
              name,
              str(params.learning_rate).replace('.', ''),
              str(params.min_split_gain).replace('.', ''),
              str(params.num_trees).replace('.', ''),
          ), 'wb'))

  fig = plt.figure()
  plt.plot(output_dict['gbt_test_losses'], label='gbt')
  plt.plot(output_dict['agbt_b_test_losses'], label='agbt_b')
  plt.plot(output_dict['agbt_train_losses_1'], label='agbt1')
  plt.plot(output_dict['agbt_train_losses_2'], label='agbt01')
  plt.plot(output_dict['agbt_train_losses_3'], label='agbt001')
  plt.yscale('log')
  plt.legend()
  fig.savefig(
      gfile.Open(
          figure_dir +
          'log_test_{:s}_lr_{:s}_min_split_gain_{:s}_num_trees_{:s}'.format(
              name,
              str(params.learning_rate).replace('.', ''),
              str(params.min_split_gain).replace('.', ''),
              str(params.num_trees).replace('.', ''),
          ), 'wb'))


def main(argv):
  del argv

  if FLAGS.data_folder is None:
    raise ValueError('Directory with downloaded datasets must be provided.')

  if FLAGS.dataset_name == 'all_datasets':
    names = ['a1a', 'w1a', 'housing']
  else:
    names = [FLAGS.dataset_name]

  if FLAGS.output_dir is None:
    raise ValueError('Output directory must be provided.')

  for name in names:
    x_train, x_test, y_train, y_test = set_up_data(FLAGS.data_folder, name)
    train_data = Dataset(x_train, y_train)
    test_data = Dataset(x_test, y_test)

    gbt_params = collections.namedtuple('gbt_params', [
        'regularizer_const', 'min_split_gain', 'max_depth', 'learning_rate',
        'num_trees', 'early_stopping_rounds', 'loss', 'use_hessian',
        'z_shrinkage_parameter'
    ])

    params = gbt_params(
        regularizer_const=FLAGS.regularizer_const,
        min_split_gain=FLAGS.min_split_gain,
        max_depth=FLAGS.max_depth,
        learning_rate=FLAGS.learning_rate,
        num_trees=FLAGS.num_trees,
        early_stopping_rounds=FLAGS.early_stopping_rounds,
        loss=FLAGS.loss,
        use_hessian=FLAGS.use_hessian,
        z_shrinkage_parameter=FLAGS.z_shrinkage_parameter)

    gbt_method = GBT(params)
    gbt_train_losses, gbt_test_losses = (
        gbt_method.train(train_data, valid_set=test_data))

    agbt_b_method = AGBTB(params)
    agbt_b_train_losses, agbt_b_test_losses = (
        agbt_b_method.train(train_data, valid_set=test_data))

    params = params._replace(z_shrinkage_parameter=0.5)
    agbt_method_1 = AGBT(params)
    agbt_train_losses_1, agbt_test_losses_1 = (
        agbt_method_1.train(train_data, valid_set=test_data))

    params = params._replace(z_shrinkage_parameter=0.3)
    agbt_method_2 = AGBT(params)
    agbt_train_losses_2, agbt_test_losses_2 = (
        agbt_method_2.train(train_data, valid_set=test_data))

    params = params._replace(z_shrinkage_parameter=0.1)
    agbt_method_3 = AGBT(params)
    agbt_train_losses_3, agbt_test_losses_3 = (
        agbt_method_3.train(train_data, valid_set=test_data))

    output_dict = {
        'gbt_train_losses': gbt_train_losses,
        'gbt_test_losses': gbt_test_losses,
        'agbt_b_train_losses': agbt_b_train_losses,
        'agbt_b_test_losses': agbt_b_test_losses,
        'agbt_train_losses_1': agbt_train_losses_1,
        'agbt_test_losses_1': agbt_test_losses_1,
        'agbt_train_losses_2': agbt_train_losses_2,
        'agbt_test_losses_2': agbt_test_losses_2,
        'agbt_train_losses_3': agbt_train_losses_3,
        'agbt_test_losses_3': agbt_test_losses_3
    }

    save_output(output_dict, name, params)
    plot_figures(output_dict, name, params)


if __name__ == '__main__':
  app.run(main)
