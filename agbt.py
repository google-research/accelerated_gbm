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
"""This is a simple implementation of accelerated gradient boosted tree."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import time

import numpy as np

import tree as Tree

NUM_QUANTILE = 100
LARGE_NUMBER = sys.maxsize

np.random.seed(0)
np.set_printoptions(threshold=np.inf)


def combine_f_with_h(ensemble_f, ensemble_h):
  assert len(ensemble_f) == 2 * len(ensemble_h) - 1
  new_models = copy.copy(ensemble_f.models)
  new_models.append(ensemble_h.models[-1])
  new_coefficients = np.append(ensemble_f.coefficients, 0)
  new_coefficients[1:len(new_coefficients):2] = (
      new_coefficients[1:len(new_coefficients):2] + ensemble_h.coefficients)
  return Tree.TreeEnsemble(new_models, new_coefficients)


class AGBT(Tree.BoostedTrees):
  """Accelerated Gradient Boosted Trees.

  Typical usage example:
    method = AGBT(params)
    method.train(train_data, valid_set=test_data)
  """

  def train(self, train_set, valid_set=None, early_stopping_rounds=5):
    ensemble_f = Tree.TreeEnsemble()
    ensemble_g = Tree.TreeEnsemble()
    ensemble_h = Tree.TreeEnsemble()
    learning_rate = self.params.learning_rate
    z_shrinkage_parameter = self.params.z_shrinkage_parameter
    best_iteration = 0
    best_val_loss = LARGE_NUMBER
    train_start_time = time.time()
    n = len(train_set.y)
    corrected_grad = np.zeros(n)  # corresponds to c^m in the paper
    learner_h_output = np.zeros(
        n)  # corresponds to b_{\tau^2}^m(X) in the paper
    train_losses = np.array([])
    val_losses = np.array([])
    train_f_output = np.zeros(len(train_set.y))
    train_g_output = np.zeros(len(train_set.y))
    train_h_output = np.zeros(len(train_set.y))
    if valid_set:
      val_f_output = np.zeros(len(valid_set.y))
      val_g_output = np.zeros(len(valid_set.y))
      val_h_output = np.zeros(len(valid_set.y))

    for iter_cnt in range(self.params.num_trees):
      iter_start_time = time.time()
      theta = 2/(iter_cnt+2)

      if ensemble_f.models:
        ensemble_g = combine_f_with_h((1-theta)*ensemble_f, theta*ensemble_h)

      train_g_output = (1-theta) * train_f_output + theta * train_h_output
      grad, hessian = self._calc_gradient_and_hessian(train_set, train_g_output)
      learner_f = self._build_learner(train_set, grad, hessian)

      ensemble_f = ensemble_g.add(learner_f, learning_rate)

      corrected_grad = (
          grad - (iter_cnt+1)/(iter_cnt+2)*(corrected_grad - learner_h_output))
      learner_h = self._build_learner(train_set, corrected_grad, hessian)
      learner_h_output = self._calc_training_data_scores(
          train_set, Tree.TreeEnsemble([learner_h], np.array([1])))

      train_f_output = self._update_output(
          train_set, learner_f, learning_rate, train_g_output)
      train_h_output = self._update_output(
          train_set, learner_h,
          z_shrinkage_parameter / theta * learning_rate, train_h_output)
      ensemble_h.append(learner_h, z_shrinkage_parameter/theta*learning_rate)
      train_loss = self._calc_loss_from_output(train_set, train_f_output)
      train_losses = np.append(train_losses, train_loss)

      if valid_set:
        val_g_output = (1-theta) * val_f_output + theta * val_h_output
        val_f_output = self._update_output(
            valid_set, learner_f, learning_rate, val_g_output)
        val_h_output = self._update_output(
            valid_set, learner_h,
            z_shrinkage_parameter / theta * learning_rate, val_h_output)
        val_loss = self._calc_loss_from_output(valid_set, val_f_output)
        val_losses = np.append(val_losses, val_loss)
        val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
      else:
        val_loss_str = ''
      print(
          "Iter {:>3}, Train's loss: {:.10f}, Valid's loss: {}, Elapsed: {:.2f} secs"
          .format(iter_cnt, train_loss, val_loss_str,
                  time.time() - iter_start_time))
      if valid_set and val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_iteration = iter_cnt
      if iter_cnt - best_iteration >= self.params.early_stopping_rounds:
        print('Early stopping, best iteration is:')
        print("Iter {:>3}, Train's loss: {:.10f}".format(
            best_iteration, best_val_loss))
        break

    self.tree_ensemble = ensemble_f
    self.best_iteration = best_iteration
    print('Training finished. Elapsed: {:.2f} secs'.
          format(time.time() - train_start_time))

    if valid_set:
      return train_losses, val_losses
    else:
      return train_losses

  def predict(self, x, ensemble=None, num_iteration=None):
    if ensemble is None:
      ensemble = self.tree_ensemble
    return ensemble.predict(x)
