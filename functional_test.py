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
"""Tests for tinygbt.functional."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import functional as F
from absl.testing import absltest


class LogisticLoss(absltest.TestCase):

  def setUp(self):
    super(LogisticLoss, self).setUp()
    self.loss = F.LogisticLoss(True)

  def testLossValue(self):
    prediction = np.array([1, -1])
    labels = np.array([1, 1])
    expected_output = np.array([np.log(1 + np.exp(-1)), np.log(1 + np.exp(1))])
    self.assertSameElements(expected_output,
                            self.loss.loss_value(prediction, labels))

  def testNegativeGradient(self):
    prediction = np.array([1, -1])
    labels = np.array([1, 1])
    expected_output = np.array(
        [np.exp(-1) / (1 + np.exp(-1)),
         np.exp(1) / (1 + np.exp(1))])
    self.assertSameElements(expected_output,
                            self.loss.negative_gradient(prediction, labels))

  def testHessian(self):
    prediction = np.array([1, -1])
    labels = np.array([1, 1])
    expected_output = np.array(
        [np.exp(-1) / (1 + np.exp(-1))**2,
         np.exp(1) / (1 + np.exp(1))**2])
    self.assertSameElements(expected_output,
                            self.loss.hessian(prediction, labels))


if __name__ == '__main__':
  absltest.main()
