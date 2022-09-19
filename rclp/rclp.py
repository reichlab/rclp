import pickle

import numpy as np
import abc

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

from . import util

class Base_RCLP(abc.ABC):
    """
    Base class for a recalibrated linear pool, with estimation by optimizing
    log score.
    """
    def __init__(self, M, rc_parameters={}) -> None:
        """
        Initialize a Base_RCLP model
        
        Parameters
        ----------
        M: integer
            number of component models
        rc_parameters: dictionary
            parameters for recalibration
        
        Returns
        -------
        None
        """
        softmax_bijector = tfb.SoftmaxCentered()
        
        w_xavier_hw = 1.0 / tf.math.sqrt(tf.constant(M-1, dtype=tf.float32))
        def init_w():
            return softmax_bijector.forward(
                tf.random.uniform((M-1,), -w_xavier_hw, w_xavier_hw))
        
        rc_parameters.update({
            'w': {
                'init': init_w,
                'bijector': softmax_bijector
            }
        })
        self._parameter_defs = rc_parameters
        
        # dictionary of transformed variables for ensemble parameters
        self.parameters = {
            param_name: tfp.util.TransformedVariable(
                initial_value=param_config['init'](),
                bijector=param_config['bijector'],
                name=param_name,
                dtype=np.float32
            ) \
                for param_name, param_config in self._parameter_defs.items()
        }
    
    
    @property
    def parameters(self):
        """
        Dictionary of ensemble model parameters
        """
        return self._parameters
    
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value
    
    
    @abc.abstractmethod
    def log_g(self, lp_F, **kwargs):
        """
        Calculate the log density of a recalibrating transformation for an
        ensemble forecast. This should correspond to a distribution with support
        on the interval [0, 1].
        
        Parameters
        ----------
        lp_F: 1D tensor with length N
            cdf values from the linear pool for observation cases i = 1, ..., N
        **kwargs: dictionary
            additional parameters as specified by the concrete implementation,
            including model parameters needed to construct the ensemble as
            returned by unpack_params
        
        Returns
        -------
        1D tensor of length N
            log recalibration density evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
    
    
    def ens_log_f(self, log_f, log_F):
        """
        Log pdf of ensemble
        
        Parameters
        ----------
        log_f: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        log_F: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        
        Returns
        -------
        Log of ensemble pdf for all observation cases as a tensor of length N
        """
        # separately extract parameters for the linear pool (just 'w')
        # and parameters for recalibration (all other than 'w')
        w = self.parameters['w']
        rc_parameters = {k: v for k,v in self.parameters.items() if k != 'w'}
        
        # adjust log_f, log_F, and w to handle missing values
        log_f, log_F, w = util.handle_missingness(log_f=log_f, log_F=log_F, w=w)
        
        # log_f and log_F for linear pool
        lp_log_f = tf.reduce_logsumexp(log_f + tf.math.log(w), axis=1)
        lp_log_F = tf.reduce_logsumexp(log_F + tf.math.log(w), axis=1)
        
        # log probability for recalibrated linear pool
        # log[f(y)] = log[ g{ \sum_{m=1}^M \pi_m F_{m,i}(y); \theta } ]
        #              + log[ \sum_{m=1}^M \pi_m f_{m,i}(y) ]
        ens_log_f = self.log_g(lp_F=tf.math.exp(lp_log_F), **rc_parameters) + \
            lp_log_f
        
        return ens_log_f
    
    
    def log_score_objective(self, log_f, log_F):
        """
        Log score objective function for use during parameter estimation
        
        Parameters
        ----------
        param_vec: 1D tensor of length self.n_param
            parameters vector
        log_f: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        log_F: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        
        Returns
        -------
        Total log score over all predictions as scalar tensor
        """
        # negative sum of ensemble log pdf values across
        # observation indices i = 1, ..., N
        return -tf.reduce_sum(self.ens_log_f(log_f, log_F))
    
    
    def fit(self,
            log_f,
            log_F,
            optim_method = "adam",
            num_iter = 100,
            learning_rate = 0.1,
            verbose = False,
            save_frequency = None,
            save_path = None):
        """
        Estimate model parameters
        
        Parameters
        ----------
        log_f: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        log_F: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        init_param_vec: optional 1D tensor of length self.n_param
            optional initial values for the weights during estimation
        optim_method: string
            optional method for optimization.  Options are "adam" or "sgd".
        num_iter: integer
            number of iterations for optimization
        learning_rate: Tensor or a floating point value.
            The learning rate
        """
        # convert inputs to float tensors
        log_f = tf.convert_to_tensor(log_f, dtype=tf.float32)
        log_F = tf.convert_to_tensor(log_F, dtype=tf.float32)
        
        # TODO: validate log_f and log_F arguments
        # self.validate_log_f_log_F(log_f, log_F)
        
        if save_frequency == None:
            save_frequency = num_iter + 1
        
        # create optimizer
        if optim_method == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optim_method == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate = learning_rate)
        
        # initiate loss trace
        lls_ = np.zeros(num_iter, np.float32)
        
        # list of trainable variables for which to calculate gradient
        trainable_variables = [self.parameters[v].trainable_variables[0] \
            for v in self.parameters.keys()]
        
        # apply gradient descent num_iter times
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                loss = self.log_score_objective(log_f=log_f, log_F=log_F)
            grads = tape.gradient(loss, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            lls_[i] = loss
            
            if verbose:
                print(i)
                print("param estimates = ")
                print(self.parameters)
                print("loss = ")
                print(loss.numpy())
                print("grads = ")
                print(grads)
            
            if (i + 1) % save_frequency == 0:
                # save parameter estimates and loss trace
                params_to_save = {
                    'param_estimates': { k: v.numpy() for k,v in self.parameters.items() },
                    'loss_trace': lls_
                }
                
                pickle.dump(params_to_save, open(str(save_path), "wb"))
        
        # set parameter estimates
        # self.set_param_estimates_vec(params_vec_var.numpy())
        self.loss_trace = lls_



class Beta_RCLP(Base_RCLP):
    def __init__(self, M) -> None:
        """
        Initialize a beta re-calibrated linear pool model
        
        Parameters
        ----------
        M: integer
            number of component models
        
        Returns
        -------
        None
        """
        softplus_bijector = tfb.Softplus()
        def init_alpha_beta():
            return softplus_bijector.forward(
                tf.random.normal((1,)))
        
        rc_parameters = {
            'alpha': {
                'init': init_alpha_beta,
                'bijector': softplus_bijector
            },
            'beta': {
                'init': init_alpha_beta,
                'bijector': softplus_bijector
            }
        }
        
        super(Beta_RCLP, self).__init__(M=M, rc_parameters=rc_parameters)
    
    
    def log_g(self, lp_F, alpha, beta):
        """
        Calculate the log density of a recalibrating transformation for an
        ensemble forecast. For a beta recalibrated linear pool, the
        recalibrating transformation corresponds to a Beta(alpha, beta)
        distribution.
        
        Parameters
        ----------
        lp_F: 1D tensor with length N
            cdf values from the linear pool for observation cases i = 1, ..., N
        alpha: scalar
            first shape parameter of Beta distribution
        beta: scalar
            second shape parameter of Beta distribution
        
        Returns
        -------
        1D tensor of length N
            log recalibration density evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
        # note that we transform lp_F away from 0 and 1 to avoid numeric issues
        # at the boundary of the support of the Beta distribution
        return tfd.Beta(alpha, beta).log_prob(lp_F * 0.99999 + 0.000005)



class LinearPool(Base_RCLP):
    def __init__(self, M) -> None:
        """
        Initialize a LinearPool model
        
        Parameters
        ----------
        M: integer
            number of component models
        
        Returns
        -------
        None
        """
        super(LinearPool, self).__init__(M=M, rc_parameters={})
    
    
    def log_g(self, lp_F):
        """
        Calculate the log density of a recalibrating transformation for an
        ensemble forecast. For a linear pool, no recalibration is done. This
        corresponds to the use of a Uniform(0, 1) distribution with log
        density that takes the value 0 everywhere.
        
        Parameters
        ----------
        lp_F: 1D tensor with length N
            cdf values from the linear pool for observation cases i = 1, ..., N
        
        Returns
        -------
        1D tensor of length N
            log recalibration density evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
        
        return tf.zeros_like(lp_F)
