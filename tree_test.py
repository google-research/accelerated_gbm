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
"""Tests for accelerated_gbm.tree."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tree
from absl.testing import absltest


class TreeNode(absltest.TestCase):

  def setUp(self):
    super(TreeNode, self).setUp()
    self.tree_node = tree.TreeNode()

  def testFindInstanceForQuantileBoundaries(self):
    sorted_values = [0, 0, 0, 1, 1, 2, 3, 3]
    num_quantile = 4
    expected_output = [-1, 2, 4, 5, 7]
    self.assertSameElements(
        expected_output,
        self.tree_node.find_instance_for_quantile_boundaries(
            sorted_values, num_quantile)
    )

  def testFindInstanceForQuantileBoundariesWhenMoreQuantiles(self):
    sorted_values = [0, 0, 0, 1, 1, 2, 3, 3]
    num_quantile = 10
    expected_output = [-1, 2, 4, 5, 7]
    self.assertSameElements(
        expected_output,
        self.tree_node.find_instance_for_quantile_boundaries(
            sorted_values, num_quantile)
    )

if __name__ == '__main__':
  absltest.main()
