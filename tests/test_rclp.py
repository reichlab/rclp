import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import unittest

from rclp.rclp import BetaMixtureRCLP


class Test_RCLP(unittest.TestCase):
    def test_validate_component_log_prob_cdf_all_correct_none_missing(self):
        log_f = tf.constant(np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))
        log_F = tf.constant(5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))

        rclp = BetaMixtureRCLP(M=1, K=1)

        # no errors
        rclp.validate_component_log_prob_cdf(log_f, log_F)
    
    
    def test_validate_component_log_prob_cdf_all_correct_with_missing(self):
        log_f_np = np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
        log_f_np[[0, 3, 3], [1, 1, 2]] = np.nan
        log_f = tf.constant(log_f_np)
        log_F_np = 5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
        log_F_np[[0, 3, 3], [1, 1, 2]] = np.nan
        log_F = tf.constant(log_F_np)

        rclp = BetaMixtureRCLP(M=1, K=1)

        # no errors
        rclp.validate_component_log_prob_cdf(log_f, log_F)
    
    
    def test_validate_component_log_prob_cdf_mismatched_missing(self):
        log_f_np = np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
        log_f_np[[0, 3, 3], [1, 1, 2]] = np.nan
        log_f = tf.constant(log_f_np)
        log_F_np = 5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3))
        log_F_np[[0, 2, 3], [1, 1, 2]] = np.nan
        log_F = tf.constant(log_F_np)

        rclp = BetaMixtureRCLP(M=1, K=1)

        # raises error
        with self.assertRaises(ValueError):
            rclp.validate_component_log_prob_cdf(log_f, log_F)
    
    
    def test_validate_component_log_prob_cdf_mismatched_shapes(self):
        log_f = tf.constant(np.linspace(1, 2 * 3, 2 * 3).reshape((2, 3)))
        log_F = tf.constant(5.0 + np.linspace(1, 5 * 3, 5 * 3).reshape((5, 3)))
        
        rclp = BetaMixtureRCLP(M=1, K=1)
        
        # raises error
        with self.assertRaises(ValueError):
            rclp.validate_component_log_prob_cdf(log_f, log_F)
    
    
    def test_validate_component_log_prob_cdf_incorrect_dim(self):
        log_f = tf.constant(np.linspace(1, 5, 5).reshape((5,)))
        log_F = tf.constant(5.0 + np.linspace(1, 5, 5).reshape((5,)))
        
        rclp = BetaMixtureRCLP(M=1, K=1)
        
        # raises error
        with self.assertRaises(ValueError):
            rclp.validate_component_log_prob_cdf(log_f, log_F)
    
    




if __name__ == '__main__':
  unittest.main()
