#!/usr/bin/python
"""
  This is gbdt implementation based on the original implementation by
  Seong-Jin Kim wtih some modificiations.
  File name: tinygbt.py
  Author: Seong-Jin Kim
  EMail: lancifollia@gmail.com
  Date created: 7/15/2018
  Reference:
    [1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
    [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision
    Tree. 2017.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

import functional as F

NUM_QUANTILE = 100
np.random.seed(0)


class Dataset(object):

  def __init__(self, x, y):
    self.x = x
    self.y = y


class TreeNode(object):
  """Tree node structure."""

  def __init__(self):
    self.is_leaf = False
    self.left_child = None
    self.right_child = None
    self.split_feature_id = None
    self.split_val = None
    self.weight = None

  def _calc_split_gain(self, gradient, h, gradient_l, h_l, gradient_r, h_r,
                       regularizer_const):
    """Loss reduction (Refer to Eq7 of Reference[1])."""

    def calc_term(gradient, h):
      return np.square(gradient) / (h + regularizer_const)

    return calc_term(gradient_l, h_l) + calc_term(gradient_r, h_r) - calc_term(
        gradient, h)

  def _calc_leaf_weight(self, grad, hessian, regularizer_const):
    """Calculate the optimal weight of this leaf node."""
    return np.sum(grad) / (np.sum(hessian) + regularizer_const)

  def find_instance_for_quantile_boundaries(self, sorted_values, num_quantile):
    """Returns the ids of instances at quantile boundary.

    Args:
      sorted_values: One feature's sorted values.
      num_quantile: number of quantile.

    Returns:
      the ids of instances that fall below quantile boundary.
    """
    n = len(sorted_values)
    linear_space = np.linspace(-1, n - 1, num_quantile + 1, dtype=int)[1:]
    quantile_id = np.zeros(num_quantile, dtype=int)
    i, j = 0, 0
    for j in range(num_quantile):
      while i < len(sorted_values) and (sorted_values[i] <=
                                        sorted_values[linear_space[j]]):
        i = i + 1
      quantile_id[j] = i - 1
    quantile_id = np.unique(quantile_id)
    quantile_id = np.append([-1], quantile_id)
    return quantile_id

  def build(self, instances, grad, hessian, depth, params):
    """Exact greedy alogirithm for split finding."""
    assert instances.shape[0] == len(grad) == len(hessian)
    if depth > params.max_depth:
      self.is_leaf = True
      self.weight = (
          self._calc_leaf_weight(grad, hessian, params.regularizer_const))
      return

    gradient = np.sum(grad)
    h = np.sum(hessian)
    best_gain = 0.
    best_feature_id = None
    best_val = 0.
    best_left_instance_ids = None
    best_right_instance_ids = None
    for feature_id in range(instances.shape[1]):
      gradient_l, h_l = 0., 0.
      sorted_values = np.sort(instances[:, feature_id])
      sorted_instance_ids = instances[:, feature_id].argsort()
      quantile_id = self.find_instance_for_quantile_boundaries(
          sorted_values, NUM_QUANTILE)
      num_quantile_id = len(quantile_id)

      for j in range(0, num_quantile_id - 1):
        gradient_l += np.sum(
            grad[sorted_instance_ids[(1 +
                                      quantile_id[j]):(1 +
                                                       quantile_id[j + 1])]])
        h_l += np.sum(
            hessian[sorted_instance_ids[(1 +
                                         quantile_id[j]):(1 +
                                                          quantile_id[j + 1])]])
        gradient_r = gradient - gradient_l
        h_r = h - h_l
        current_gain = (
            self._calc_split_gain(gradient, h, gradient_l, h_l, gradient_r, h_r,
                                  params.regularizer_const))
        if current_gain > best_gain:
          best_gain = current_gain
          best_feature_id = feature_id
          best_val = instances[sorted_instance_ids[quantile_id[j +
                                                               1]]][feature_id]
          best_left_instance_ids = sorted_instance_ids[:quantile_id[j + 1] + 1]
          best_right_instance_ids = sorted_instance_ids[quantile_id[j + 1] + 1:]

    if best_gain < params.min_split_gain:
      self.is_leaf = True
      self.weight = self._calc_leaf_weight(grad, hessian,
                                           params.regularizer_const)
    else:
      self.split_feature_id = best_feature_id
      self.split_val = best_val

      self.left_child = TreeNode()
      self.left_child.build(instances[best_left_instance_ids],
                            grad[best_left_instance_ids],
                            hessian[best_left_instance_ids], depth + 1, params)

      self.right_child = TreeNode()
      self.right_child.build(instances[best_right_instance_ids],
                             grad[best_right_instance_ids],
                             hessian[best_right_instance_ids], depth + 1,
                             params)

  def predict(self, x):
    if self.is_leaf:
      return self.weight
    else:
      if x[self.split_feature_id] <= self.split_val:
        return self.left_child.predict(x)
      else:
        return self.right_child.predict(x)


class Tree(object):
  """Classification and regression tree."""

  def __init__(self):
    self.root = None

  def build(self, instances, grad, hessian, params):
    assert len(instances) == len(grad) == len(hessian)
    self.root = TreeNode()
    current_depth = 0
    self.root.build(instances, grad, hessian, current_depth, params)

  def predict(self, x):
    return self.root.predict(x)


class TreeEnsemble(object):
  """Ensemble of classification and regression tree."""

  def __init__(self, models=None, coefficients=np.array([])):
    if not models:
      self.models = []
    else:
      self.models = models
    self.coefficients = coefficients

  def __rmul__(self, multiplier):
    new_models = copy.copy(self.models)
    new_coefficients = self.coefficients * multiplier
    return TreeEnsemble(new_models, new_coefficients)

  def __add__(self, other):
    total_models = self.models + other.models
    total_coefficients = np.append(self.coefficients, other.coefficients)
    return TreeEnsemble(total_models, total_coefficients)

  def __len__(self):
    assert len(self.models) == len(self.coefficients)
    return len(self.models)

  def append(self, learner, multiplier=1):
    self.models.append(learner)
    self.coefficients = np.append(self.coefficients, multiplier)

  def add(self, learner, multiplier):
    new_models = copy.copy(self.models)
    new_models.append(learner)
    new_coefficients = np.append(self.coefficients, multiplier)
    return TreeEnsemble(new_models, new_coefficients)

  def predict(self, x, num_trees=None):
    if not self.models:
      return 0
    else:
      if num_trees is None:
        num_trees = len(self.models)
      return np.sum(self.coefficients[i] * self.models[i].predict(x)
                    for i in range(num_trees))


class BoostedTrees(BaseEstimator, ClassifierMixin):
  """Class of boosted trees.

  This is a super-class used in GBT, AGBT, and AGBT_B.
  """

  def __init__(self,
               params,
               max_depth=None,
               learning_rate=None,
               min_split_gain=None,
               z_shrinkage_parameter=None,
               num_trees=None):
    self.tree_ensemble = TreeEnsemble()
    self.params = params
    self.best_iteration = 0
    if params.loss == "L2Loss":
      self.loss = F.L2Loss(params.use_hessian)
    elif params.loss == "LogisticLoss":
      self.loss = F.LogisticLoss(params.use_hessian)

  def _calc_training_data_output(self, train_set, tree_ensemble):
    if not tree_ensemble.models:
      return np.zeros(len(train_set.y))
    x = train_set.x
    output = np.zeros(len(x))
    for i in range(len(x)):
      output[i] = tree_ensemble.predict(x[i])
    return output

  def _calc_gradient_and_hessian(self, train_set, output):
    return (self.loss.negative_gradient(output, train_set.y),
            self.loss.hessian(output, train_set.y))

  def _calc_loss(self, tree_ensemble, data_set):
    """For now, only L2 loss and Logistic loss are supported."""
    predict = []
    for x in data_set.x:
      predict.append(tree_ensemble.predict(x))
    return np.mean(self.loss.loss_value(np.array(predict), data_set.y))

  def _build_learner(self, train_set, grad, hessian):
    learner = Tree()
    learner.build(train_set.x, grad, hessian, self.params)
    return learner

  def _update_output(self, data_set, learner, coefficient, output):
    x = data_set.x
    new_output = np.copy(output)
    for i in range(len(x)):
      new_output[i] += coefficient * learner.predict(x[i])
    return new_output

  def _calc_loss_from_output(self, data_set, output):
    return np.mean(self.loss.loss_value(output, data_set.y))

  def _calc_training_data_scores(self, train_set, ensemble):
    if not ensemble.models:
      return np.zeros(len(train_set.y))
    x = train_set.x
    scores = np.zeros(len(x))
    for i in range(len(x)):
      scores[i] = ensemble.predict(x[i])
    return scores

  def fit(self, x, y=None):
    train_set = Dataset(x, y)
    self.train(train_set)

  def score(self, x, y):
    return -self._calc_loss(self.tree_ensemble, Dataset(x, y))

  def get_params(self, deep=True):
    return {
        "params": self.params,
        "max_depth": self.params.max_depth,
        "learning_rate": self.params.learning_rate,
        "min_split_gain": self.params.min_split_gain,
        "z_shrinkage_parameter": self.params.z_shrinkage_parameter,
        "num_trees": self.params.num_trees
    }

  def set_params(self, **parameters):
    print(parameters)
    for key, value in parameters.items():
      if key == "max_depth":
        self.params = self.params._replace(max_depth=value)
      if key == "learning_rate":
        self.params = self.params._replace(learning_rate=value)
      if key == "min_split_gain":
        self.params = self.params._replace(min_split_gain=value)
      if key == "z_shrinkage_parameter":
        self.params = self.params._replace(z_shrinkage_parameter=value)
      if key == "num_trees":
        self.params = self.params._replace(num_trees=value)
    return self
