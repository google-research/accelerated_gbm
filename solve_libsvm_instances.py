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
"""Run tests with LIBSVM dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

from agbt import AGBT
from agbt_b import AGBTB
import functional as F
from gbt import GBT
from tree import Dataset
from tensorflow.python.platform import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string("data_folder", None, "The directory of datasets.")

flags.DEFINE_enum("dataset_name", "all_datasets",
                  ["all_datasets", "a1a", "w1a", "housing"],
                  ("The name of instances."
                   "`all_datasets` means all of the instances in the folder."))

flags.DEFINE_enum("loss", "L2Loss", ["L2Loss", "LogisticLoss"],
                  "The loss function.")

flags.DEFINE_enum(
    "method", "AGBT", ["GBT", "AGBT", "AGBTB"],
    ("The method to use. GBT is the standard gradient boosted tree. AGBT is our"
     "proposed method and AGBTB is the method proposed by Biau et al."))

flags.DEFINE_integer(
    "early_stopping_rounds", 100000,
    ("Stop the algorithm if the validation loss does not improve after this"
     "number of iterations."))

flags.DEFINE_float(
    "z_shrinkage_parameter", 0.1,
    "The shrinkage parameter in the z-update in accelerated method.")

flags.DEFINE_integer("max_depth", 3, "Maximal depth of a tree.")
flags.DEFINE_integer("num_trees", 20, "Number of boosting iterations.")
flags.DEFINE_float("min_split_gain", 0.1, "Minimal gain for splitting a leaf.")
flags.DEFINE_float("learning_rate", 0.3, "Learning rate.")
flags.DEFINE_float("regularizer_const", 1, "Regularizer constant.")
flags.DEFINE_boolean("use_hessian", False, "Whether to use Hessian.")

TEST_SIZE = 0.2
RANDOM_STATE = 40

LOSS = {"L2Loss": F.L2Loss, "LogisticLoss": F.LogisticLoss}


def SetupData(data_folder, dataset_name):
  path = os.path.join(data_folder, dataset_name + ".txt")
  data = sklearn.datasets.load_svmlight_file(gfile.Open(path, mode="rb"))
  x = np.asarray(data[0].todense())
  y = np.array(data[1])
  return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def main(argv):
  del argv

  if FLAGS.data_folder is None:
    raise ValueError("Directory with downloaded datasets must be provided.")

  if FLAGS.dataset_name == "all_datasets":
    names = ["a1a", "w1a", "housing"]
  else:
    names = [FLAGS.dataset_name]

  for name in names:
    x_train, x_test, y_train, y_test = SetupData(FLAGS.data_folder, name)
    train_data = Dataset(x_train, y_train)
    test_data = Dataset(x_test, y_test)

    GBTParams = collections.namedtuple("GBTParams", [
        "regularizer_const", "min_split_gain", "max_depth", "learning_rate",
        "num_trees", "early_stopping_rounds", "loss", "use_hessian",
        "z_shrinkage_parameter"
    ])

    params = GBTParams(
        regularizer_const=FLAGS.regularizer_const,
        min_split_gain=FLAGS.min_split_gain,
        max_depth=FLAGS.max_depth,
        learning_rate=FLAGS.learning_rate,
        num_trees=FLAGS.num_trees,
        early_stopping_rounds=FLAGS.early_stopping_rounds,
        loss=FLAGS.loss,
        use_hessian=FLAGS.use_hessian,
        z_shrinkage_parameter=FLAGS.z_shrinkage_parameter)

    if FLAGS.method == "GBT":
      print("Start training using GBT...")
      method = GBT(params)
    elif FLAGS.method == "AGBT":
      print("Start training using AGBT...")
      method = AGBT(params)
    elif FLAGS.method == "AGBTB":
      print("Start training using AGBTB...")
      method = AGBTB(params)

    method.train(train_data, valid_set=test_data)

    print("Start predicting...")
    y_pred = []
    for x in x_test:
      y_pred.append(method.predict(x, num_iteration=method.best_iteration))

    if params.loss == "L2Loss":
      loss = F.L2Loss(params.use_hessian)
    elif params.loss == "LogisticLoss":
      loss = F.LogisticLoss(params.use_hessian)

    print("The mean loss of prediction is:",
          np.mean(loss.loss_value(np.array(y_pred), np.array(y_test))))


if __name__ == "__main__":
  app.run(main)
