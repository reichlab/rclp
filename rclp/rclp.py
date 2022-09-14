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
    @property
    @abc.abstractmethod
    def n_param(self):
        """
        Abstract property for number of free parameters of an ensemble model
        """
        pass
    
    
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
            Dictionary with parameters
        """
    
    
    @abc.abstractmethod
    def ens_log_f(self, log_f, log_F):
        """
        Calculate the log density of an ensemble forecast
        
        Parameters
        ----------
        log_f: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        log_F: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        additional parameters as specified by the concrete implementation,
            including model parameters needed to construct the ensemble as
            returned by unpack_params
        
        Returns
        -------
        ens_log_f: 1D tensor of length N
            Ensemble log pdf value for each observation case i = 1, ..., N
        """
    
    
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
        ens_log_f = self.ens_log_f(log_f=log_f, log_F=log_F,
                                   **self.unpack_params(param_vec))
        
        return -tf.reduce_sum(ens_log_f)
    
    
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
            optim_method,
            num_iter,
            learning_rate,
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
            # create a 0 vector of length equal to the number of free parameters
            init_param_vec = tf.constant(np.zeros(self.n_param), dtype=tf.float32)
        
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

        # apply gradient descent with num_iter times
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
    
    
    @property
    def n_param(self):
        """
        Set number of parameters based on kernel and number of features
        """
        return self._n_param
    
    
    @n_param.setter
    def n_param(self, value):
        self._n_param = value
    
    
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
    
    
    def ens_log_f(self, log_f, log_F, w):
        """
        Calculate the log density of an ensemble forecast
        
        Parameters
        ----------
        log_f: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        log_F: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        w: 2D tensor with shape (N, M)
            Component model weights, where `w[i, k]` is the weight given to
            model m for observation case i
        kwargs: ignored keyword arguments (in particular, log_F is not used)
        
        Returns
        -------
        ens_log_f: 1D tensor of length N
            Ensemble log pdf value for each observation case i = 1, ..., N
        """
        # adjust w and q to handle missing values
        log_f, log_F, w = util.handle_missingness(log_f=log_f, log_F=log_F, w=w)
        
        return tf.reduce_logsumexp(log_f + tf.math.log(w), axis=1)
        


