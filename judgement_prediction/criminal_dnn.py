from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os
 
import numpy as np
import tensorflow as tf
 
# Data sets
IRIS_TRAINING = "D:/案件数据/故意杀人案/train.csv"
 
IRIS_TEST = "D:/案件数据/故意杀人案/test.csv"
 
def main():
 
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float)

  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
 
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y
 
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]
 
  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[30, 20, 50,20],
                                              n_classes=5)
 
  classifier.fit(input_fn=get_train_inputs, steps=20000)
 
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]
 
  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
 

main()
