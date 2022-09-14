import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import unittest

from rclp import util


class Test_Util(unittest.TestCase):
  def test_handle_missingness_none_missing(self):
    w = tf.constant([0.1, 0.6, 0.3], dtype = "float32")
    log_f = tf.constant(np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))
    log_F = tf.constant(5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))

    result_log_f, result_log_F, result_w = util.handle_missingness(log_f, log_F, w)

    # 5 copies of the original w
    for i in range(5):
      self.assertTrue(np.all(w.numpy() == result_w.numpy()[i, ...]))
    
    # log_f and log_F are unchanged
    for i in range(5):
      self.assertTrue(np.all(log_f.numpy()[i,...] == result_log_f.numpy()[i, ...]))
      self.assertTrue(np.all(log_F.numpy()[i,...] == result_log_F.numpy()[i, ...]))

    # Assert sum of weights across models are all approximately 1
    self.assertTrue(
      np.all(np.abs(tf.math.reduce_sum(result_w, axis = -1).numpy() - np.ones((5, ))) < 1e-7))


  def test_handle_missingness_with_missing(self):
    w = tf.constant([0.1, 0.6, 0.3], dtype = "float32")
    w_np = w.numpy()
    log_f_np = np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
    log_f_np[[0, 3, 3], [1, 1, 2]] = np.nan
    log_f = tf.constant(log_f_np)
    log_F_np = 5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
    log_F_np[[0, 3, 3], [1, 1, 2]] = np.nan
    log_F = tf.constant(log_F_np)

    result_log_f, result_log_F, result_w = util.handle_missingness(log_f, log_F, w)

    # entries at indices i with no missingness are copies of the original w
    for i in [1,2,4]:
      self.assertTrue(
        np.all(np.abs(w.numpy() - result_w.numpy()[i, ...]) < 1e-7))

    # in rows i with no missingness, log_f and log_F are unchanged
    for i in [1,2,4]:
      self.assertTrue(np.all(log_f.numpy()[i,...] == result_log_f.numpy()[i, ...]))
      self.assertTrue(np.all(log_F.numpy()[i,...] == result_log_F.numpy()[i, ...]))

    # for values of i with missingness,
    # entries [i, m] with missingness are 0
    self.assertTrue(
      np.all(result_w.numpy()[[0, 3, 3], [1, 1, 2]] == np.zeros(3)))
    
    self.assertTrue(
      np.all(result_log_f.numpy()[[0, 3, 3], [1, 1, 2]] == np.zeros(3)))
    
    self.assertTrue(
      np.all(result_log_F.numpy()[[0, 3, 3], [1, 1, 2]] == np.zeros(3)))

    # for rows (i, :) with missingness, entries at non-missing points are
    # proportional to original weights
    self.assertTrue(
      np.all(result_w.numpy()[0, [0, 2]] == w_np[[0, 2]] / np.sum(w_np[[0, 2]])))
    self.assertTrue(
      np.all(result_w.numpy()[3, 0] == 1.0))

    # for rows i with missingness, entries of log_f and log_F at non-missing
    # points are unchanged
    self.assertTrue(
      np.all(result_log_f.numpy()[0, [0, 2]] == log_f_np[0, [0, 2]]))
    self.assertTrue(
      np.all(result_log_F.numpy()[0, [0, 2]] == log_F_np[0, [0, 2]]))
    self.assertTrue(
      np.all(result_log_f.numpy()[3, 0] == log_f_np[3, 0]))
    self.assertTrue(
      np.all(result_log_F.numpy()[3, 0] == log_F_np[3, 0]))

    # Assert sum of weights across models are all approximately 1
    self.assertTrue(
      np.all(np.abs(tf.math.reduce_sum(result_w, axis = -1).numpy() - np.ones((5, ))) < 1e-7))


#   def test_unpack_params(self):
#     tau_groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
#     param_vec = tf.constant(np.log([2,1,3,2,4,3]), dtype = "float64")
#     params_dict = qens.MeanQEns().unpack_params(
#       param_vec=param_vec,
#       M=3,
#       tau_groups=tau_groups
#     )
#     w = params_dict['w']

#     # Assert w has shape (K, M) == (10, 3)
#     self.assertTrue(np.all(tf.shape(w).numpy() == np.array([10, 3])))

#     # Assert row sums are all approximately 1
#     self.assertTrue(
#       np.all(np.abs(tf.math.reduce_sum(w, axis = 1).numpy() - np.ones(10)) < 1e-7))

#     # Assert rows defined by tau_groups are equal to each other
#     for i in [1,2]:
#       self.assertTrue(np.all(w[0, :].numpy() == w[i, :].numpy()))

#     for i in [4,5,6]:
#       self.assertTrue(np.all(w[3, :].numpy() == w[i, :].numpy()))

#     for i in [8,9]:
#       self.assertTrue(np.all(w[7, :].numpy() == w[i, :].numpy()))


#   def test_pinball_loss(self):
#     y = np.array([0.1, 0.2, 0.3])
#     q = np.concatenate((
#         np.array([0.2, -0.5, 0.4]).reshape(3,1), 
#         np.array([0.1, 0.7, -0.5]).reshape(3,1)), axis = 1)
#     tau = np.array([0.2, 0.5])

#     actual = qens.MeanQEns().pinball_loss(tf.constant(y), tf.constant(q), tf.constant(tau))

#     # calculate expected 
#     expected = 0
#     for i in range(q.shape[0]):
#         for k in range(q.shape[1]):
#             if y[i] < q[i,k]:
#                 expected += (1 - tau[k]) * (q[i,k] - y[i])
#             else:
#                 expected += (0 - tau[k]) * (q[i,k] - y[i])
#     expected = expected / 6.

#     self.assertAlmostEqual(actual.numpy(),expected, places=7)

if __name__ == '__main__':
  unittest.main()
