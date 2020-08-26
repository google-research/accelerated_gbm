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
"""The util functions used in the experiments."""

from __future__ import division
import numpy as np


class L2Loss(object):
  """L2 loss function.

  use_hessian corresponds to whether we isuse hessian or a constant upper
  bound of the hessian.

  For L2 loss, the hessian is always 1.
  """

  def __init__(self, use_hessian):
    self.use_hessian = use_hessian

  def loss_value(self, prediction, labels):
    return 1 / 2 * (prediction - labels)**2

  def negative_gradient(self, prediction, labels):
    return labels - prediction

  def hessian(self, prediction, labels):
    del prediction
    return np.ones(len(labels))


class LogisticLoss(object):
  """Logistic loss function.

  use_hessian corresponds to whether we use hessian or a constant upper bound of
  the hessian. The labels are either -1 or 1.
  For logistic loss, the upper bound of hessian is 1/4.
  """

  def __init__(self, use_hessian):
    self.use_hessian = use_hessian

  def loss_value(self, prediction, labels):
    # labels are -1 and 1.
    return np.log(1 + np.exp(np.nan_to_num(-prediction * labels)))

  def negative_gradient(self, prediction, labels):

    temp = np.nan_to_num(np.exp(-labels * prediction))
    return labels * temp / (1 + temp)

  def hessian(self, prediction, labels):
    if self.use_hessian:
      temp = np.nan_to_num(np.exp(-labels * prediction))
      return temp / (1 + temp)**2
    else:
      # return a scalar upper bound of the hessian
      return np.ones(len(labels))
