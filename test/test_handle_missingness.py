import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import unittest

from rclp.rclp import BetaMixtureRCLP


class Test_HandleMissingness(unittest.TestCase):
  def test_handle_missingness_none_missing(self):
    rclp = BetaMixtureRCLP(M=1, K=1)
    w = tf.constant([0.1, 0.6, 0.3], dtype = "float32")
    rclp.parameters = {'lp_w': w}
    
    log_f = tf.constant(np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))
    log_F = tf.constant(5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))

    result_log_f, result_log_F, result_w = rclp.handle_missingness(log_f, log_F)

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
    rclp = BetaMixtureRCLP(M=1, K=1)
    w = tf.constant([0.1, 0.6, 0.3], dtype = "float32")
    w_np = w.numpy()
    rclp.parameters = {'lp_w': w}
    
    log_f_np = np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
    log_f_np[[0, 3, 3], [1, 1, 2]] = np.nan
    log_f = tf.constant(log_f_np)
    log_F_np = 5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
    log_F_np[[0, 3, 3], [1, 1, 2]] = np.nan
    log_F = tf.constant(log_F_np)

    result_log_f, result_log_F, result_w = rclp.handle_missingness(log_f, log_F)

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
      np.all(result_log_f.numpy()[[0, 3, 3], [1, 1, 2]] == np.full((3,), -np.inf)))
    
    self.assertTrue(
      np.all(result_log_F.numpy()[[0, 3, 3], [1, 1, 2]] == np.full((3,), -np.inf)))

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


if __name__ == '__main__':
  unittest.main()
