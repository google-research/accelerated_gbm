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
"""GBT is a simple implementation of gradient boosted tree."""

from __future__ import absolute_import
from __future__ import division

import sys
import time

import numpy as np

import tree as Tree

NUM_QUANTILE = 100
np.random.seed(0)
LARGE_NUMBER = sys.maxsize


class GBT(Tree.BoostedTrees):
  """Simple implementation of Gradient Boosted Trees.

  Typical usage example:
    method = GBT(params)
    method.train(train_data, valid_set=test_data)
  """

  def train(self, train_set, valid_set=None, early_stopping_rounds=5):
    tree_ensemble = Tree.TreeEnsemble()
    learning_rate = self.params.learning_rate
    best_iteration = 0
    best_val_loss = LARGE_NUMBER
    train_start_time = time.time()
    train_losses = np.array([])
    train_output = np.zeros(len(train_set.y))
    if valid_set:
      val_losses = np.array([])
      val_output = np.zeros(len(valid_set.y))

    for iter_cnt in range(self.params.num_trees):
      iter_start_time = time.time()

      grad, hessian = self._calc_gradient_and_hessian(train_set, train_output)
      learner = self._build_learner(train_set, grad, hessian)
      tree_ensemble.append(learner, learning_rate)
      train_output = self._update_output(train_set, learner, learning_rate,
                                         train_output)

      train_loss = self._calc_loss_from_output(train_set, train_output)
      train_losses = np.append(train_losses, train_loss)

      if valid_set:
        val_output = self._update_output(valid_set, learner, learning_rate,
                                         val_output)
        val_loss = self._calc_loss_from_output(valid_set, val_output)
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

    self.tree_ensemble = tree_ensemble
    self.best_iteration = best_iteration
    print('Training finished. Elapsed: {:.2f} secs'.format(time.time() -
                                                           train_start_time))

    if valid_set:
      return train_losses, val_losses
    else:
      return train_losses

  def predict(self, x, tree_ensemble=None, num_iteration=None):
    if tree_ensemble is None:
      tree_ensemble = self.tree_ensemble
    return tree_ensemble.predict(x)
