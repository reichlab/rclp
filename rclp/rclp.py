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
    @abc.abstractmethod
    def unpack_params(self, param_vec):
        """
        Convert from a vector of parameters to a dictionary of parameter values
        suitable for use in a call to self.ens_log_f
        
        Parameters
        ----------
        param_vec: 1D tensor of length self.n_param
            vector of parameters on the scale of unconstrained real numbers
        
        Returns
        -------
        params_dict: dictionary
            Dictionary with parameters. At minimum, includes the weights 'w' for
            the linear pool
        """
    
    
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
    
    
    @property
    def n_param(self):
        """
        Set number of parameters based on kernel and number of features
        """
        return self._n_param
    
    
    @n_param.setter
    def n_param(self, value):
        self._n_param = value
    
    
    def ens_log_f(self, param_vec, log_f, log_F):
        """
        Log pdf of ensemble
        
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
        Log of ensemble pdf for all observation cases as a tensor of length N
        """
        # unpack parameters from vector of real numbers to dictionary of
        # appropriately constrained parameter values
        param_dict = self.unpack_params(param_vec)
        w = param_dict.pop('w')
        
        # adjust log_f, log_F, and w to handle missing values
        log_f, log_F, w = util.handle_missingness(log_f=log_f, log_F=log_F, w=w)
        
        # log_f and log_F for linear pool
        lp_log_f = tf.reduce_logsumexp(log_f + tf.math.log(w), axis=1)
        lp_log_F = tf.reduce_logsumexp(log_F + tf.math.log(w), axis=1)
        
        # log probability for recalibrated linear pool
        # log[f(y)] = log[ g{ \sum_{m=1}^M \pi_m F_{m,i}(y); \theta } ]
        #              + log[ \sum_{m=1}^M \pi_m f_{m,i}(y) ]
        ens_log_f = self.log_g(lp_F=tf.math.exp(lp_log_F), **param_dict) + lp_log_f
        
        return ens_log_f
    
    
    def log_score_objective(self, param_vec, log_f, log_F):
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
        # sum ensemble log pdf values across observation indices i = 1, ..., N
        return -tf.reduce_sum(self.ens_log_f(param_vec, log_f, log_F))
    
    
    def get_param_estimates_vec(self):
        """
        Get parameter estimates in vector form 
        
        Returns
        ----------
        param_estimates_vec: 1D tensor of length self.n_param
        """
        return self._param_estimates_vec
    
    
    def set_param_estimates_vec(self, param_estimates_vec):
        """
        Set parameter estimates in vector form
        
        Parameters
        ----------
        param_estimates_vec: 1D tensor of length self.n_param
        """
        self._param_estimates_vec = param_estimates_vec
    
    
    def fit(self,
            log_f,
            log_F,
            optim_method = "adam",
            num_iter = 100,
            learning_rate = 0.1,
            init_param_vec = None,
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
        
        if init_param_vec == None:
            # Xavier weight initialization
            # weight ~ Unif(-1/sqrt(n_param), 1/sqrt(n_param))
            xavier_half_width = 1.0 / tf.math.sqrt(tf.constant(self.n_param,
                                                               dtype=tf.float32))
            init_param_vec = tf.random.uniform((self.n_param,),
                                               -xavier_half_width,
                                               xavier_half_width)
        
        # declare variable representing parameters to estimate
        params_vec_var = tf.Variable(
            initial_value=init_param_vec,
            name='params_vec',
            dtype=np.float32)
        
        # create optimizer
        if optim_method == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optim_method == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate = learning_rate)
        
        # initiate loss trace
        lls_ = np.zeros(num_iter, np.float32)
        
        # create a list of trainable variables
        trainable_variables = [params_vec_var]

        # apply gradient descent num_iter times
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                loss = self.log_score_objective(param_vec=params_vec_var,
                                                log_f=log_f, log_F=log_F)
            grads = tape.gradient(loss, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            lls_[i] = loss

            if verbose:
                print(i)
                print("param estimates vec = ")
                print(params_vec_var.numpy())
                print("loss = ")
                print(loss.numpy())
                print("grads = ")
                print(grads)
            
            if (i + 1) % save_frequency == 0:
                # save parameter estimates and loss trace
                params_to_save = {
                    'param_estimates_vec': params_vec_var.numpy(),
                    'loss_trace': lls_
                }
    
                pickle.dump(params_to_save, open(str(save_path), "wb"))

        # set parameter estimates
        self.set_param_estimates_vec(params_vec_var.numpy())
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
        self.n_param = int(M - 1 + 2)
        
        super(Beta_RCLP, self).__init__()
    
    
    def unpack_params(self, param_vec):
        """
        Convert from a vector of parameters to a dictionary of parameter values
        suitable for use in a call to self.ens_log_f
        
        Parameters
        ----------
        param_vec: 1D tensor of length self.n_param
            vector of parameters on the scale of unconstrained real numbers
        
        Returns
        -------
        params_dict: dictionary
            Dictionary with parameters
        """
        return {
            'w': tfb.SoftmaxCentered().forward(param_vec[:-2]),
            'alpha': tfb.Softplus().forward(param_vec[-2]),
            'beta': tfb.Softplus().forward(param_vec[-1])
        }
    
    
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
        self.n_param = int(M - 1)
        
        super(LinearPool, self).__init__()
    
    
    def unpack_params(self, param_vec):
        """
        Convert from a vector of parameters to a dictionary of parameter values
        suitable for use in a call to self.ens_log_f
        
        Parameters
        ----------
        param_vec: 1D tensor of length self.n_param
            vector of parameters on the scale of unconstrained real numbers
        
        Returns
        -------
        params_dict: dictionary
            Dictionary with parameters
        """
        return { 'w': tfb.SoftmaxCentered().forward(param_vec) }
    
    
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


