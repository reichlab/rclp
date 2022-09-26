import pickle

import numpy as np
import abc

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

class BaseRCLP(abc.ABC):
    """
    Base class for a recalibrated linear pool, with estimation by optimizing
    log score.
    """
    def __init__(self, M, rc_parameters={}) -> None:
        """
        Initialize an RCLP model
        
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
        
        # Xavier initialization for w
        w_xavier_hw = 1.0 / tf.math.sqrt(tf.constant(M-1, dtype=tf.float32))
        def init_lp_w():
            return softmax_bijector.forward(
                tf.random.uniform((M-1,), -w_xavier_hw, w_xavier_hw))
        
        rc_parameters.update({
            'lp_w': {
                'init': init_lp_w,
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
        
        # initialize loss trace
        self.loss_trace = np.zeros((0,), np.float32)
    
    
    @property
    def parameters(self):
        """
        Dictionary of ensemble model parameters
        """
        return self._parameters
    
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value
    
    
    @property
    def loss_trace(self):
        """
        Dictionary of ensemble model parameters
        """
        return self._loss_trace
    
    
    @loss_trace.setter
    def loss_trace(self, value):
        self._loss_trace = value
    
    
    def handle_missingness(self, component_log_prob, component_log_cdf,
                           validate=True):
        """
        Broadcast w to the same shape as component_log_prob, set weights
        corresponding to missing entries of component_log_prob to 0, and
        re-normalize so that the weights sum to 1. Then, replace nans in
        component_log_prob and component_log_cdf with 0. It is implicitly
        assumed, but not explicitly checked, that component_log_prob and
        component_log_cdf have missing values in the same locations.

        Parameters
        ----------
        component_log_prob: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        validate: boolean
            Indicator of whether component_log_prob and component_log_cdf should
            be validated
        
        Returns
        -------
        component_log_prob_nans_replaced: 2D tensor with shape (N, M)
            Component log-pdf values with nans replaced with -infinity
        component_log_cdf_nans_replaced: 2D tensor with shape (N, M)
            Component log-cdf values with nans replaced with -infinity
        broadcast_w: 2D tensor with shape (N, M)
            broadcast_w has N copies of the argument w with weights w[i,m] set
            to 0 at indices where log_f[i,m] is nan. The weights are then
            re-normalized to sum to 1 within each combination of i.
        """
        # if necessary, validate
        if validate:
            self.validate_component_log_prob_cdf(component_log_prob,
                                                 component_log_cdf)
        
        # extract weights for linear pool
        w = self.parameters['lp_w']
        
        # broadcast w to the same shape as log_f, creating N copies of w
        log_prob_shape = tf.shape(component_log_prob).numpy()
        broadcast_w = tf.broadcast_to(w, log_prob_shape)

        # if there is missingness, adjust entries of broadcast_w
        missing_mask = tf.math.is_nan(component_log_prob)
        if tf.reduce_any(missing_mask):
            # nonmissing mask has shape (N, M), with entries
            # 0 where log_f had missing values and 1 elsewhere
            nonmissing_mask = tf.cast(
                tf.logical_not(missing_mask),
                dtype = broadcast_w.dtype)

            # set weights corresponding to missing entries of q to 0
            broadcast_w = tf.math.multiply(broadcast_w, nonmissing_mask)

            # renormalize weights to sum to 1 along the model axis
            (broadcast_w, _) = tf.linalg.normalize(broadcast_w, ord = 1, axis = -1)

        # replace nan with 0 in log_f and log_F
        component_log_prob_nans_replaced = tf.where(missing_mask,
                                                    np.float32(-np.inf),
                                                    np.float32(component_log_prob))
        component_log_cdf_nans_replaced = tf.where(missing_mask,
                                                np.float32(-np.inf),
                                                np.float32(component_log_cdf))
        
        return component_log_prob_nans_replaced, component_log_cdf_nans_replaced, broadcast_w
    
    
    def validate_component_log_prob_cdf(self,
                                        component_log_prob, component_log_cdf):
        """
        Validate that component_log_prob and component_log_cdf they have the
        same shape of dimension 2 and have the same patterns of missingness.
        
        Parameters
        ----------
        component_log_prob: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        
        Returns
        -------
        None
        """
        if not component_log_prob.shape == component_log_cdf.shape:
            raise ValueError("component_log_prob and component_log_cdf must have the same shape")
        
        if not len(component_log_prob.shape) == 2:
            raise ValueError("component_log_prob and component_log_cdf must have be 2-dimensional")
        
        prob_missing_mask = tf.math.is_nan(component_log_prob)
        cdf_missing_mask = tf.math.is_nan(component_log_cdf)
        if not tf.reduce_all(prob_missing_mask == cdf_missing_mask):
            raise ValueError("any missing values in component_log_prob and component_log_cdf must be at the same indices")
    
    
    @abc.abstractmethod
    def rc_log_prob(self, lp_log_cdf, **kwargs):
        """
        Calculate the log density of a recalibrating transformation for an
        ensemble forecast. This should correspond to a distribution with support
        on the interval [0, 1].
        
        Parameters
        ----------
        lp_log_cdf: 1D tensor with length N
            log cdf values from the linear pool for observation cases i = 1, ..., N
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
    
    
    @abc.abstractmethod
    def rc_log_cdf(self, lp_log_cdf, **kwargs):
        """
        Calculate the log cdf of a recalibrating transformation for an
        ensemble forecast. This should correspond to a distribution with support
        on the interval [0, 1].
        
        Parameters
        ----------
        lp_log_cdf: 1D tensor with length N
            log cdf values from the linear pool for observation cases i = 1, ..., N
        **kwargs: dictionary
            additional parameters as specified by the concrete implementation,
            including model parameters needed to construct the ensemble as
            returned by unpack_params
        
        Returns
        -------
        1D tensor of length N
            log recalibration cdf evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
    
    
    def log_prob(self, component_log_prob, component_log_cdf, validate=True):
        """
        Log pdf of recalibrated linear pool ensemble
        
        Parameters
        ----------
        component_log_prob: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        validate: boolean
            Indicator of whether component_log_prob and component_log_cdf should
            be validated
        
        Returns
        -------
        Log of ensemble pdf for all observation cases as a tensor of length N
        """
        # if necessary, validate
        if validate:
            self.validate_component_log_prob_cdf(component_log_prob,
                                                 component_log_cdf)
        
        # handle missing values; returns linear pool weights and updated
        # component model probabilities
        component_log_prob, component_log_cdf, lp_w = self.handle_missingness(
            component_log_prob=component_log_prob,
            component_log_cdf=component_log_cdf,
            validate=False)
        
        # log_prob and log_cdf for linear pool
        lp_log_prob = tf.reduce_logsumexp(component_log_prob + tf.math.log(lp_w),
                                          axis=1)
        lp_log_cdf = tf.reduce_logsumexp(component_log_cdf + tf.math.log(lp_w),
                                          axis=1)
        
        # extract parameters for recalibration (all other than 'lp_w')
        rc_parameters = {k: v for k,v in self.parameters.items() if k != 'lp_w'}
        
        # log probability for recalibrated linear pool
        # log[f(y)] = log[ g{ \sum_{m=1}^M \pi_m F_{m,i}(y); \theta } ]
        #              + log[ \sum_{m=1}^M \pi_m f_{m,i}(y) ]
        log_prob = self.rc_log_prob(lp_log_cdf=lp_log_cdf,
                                    **rc_parameters) + \
            lp_log_prob
        
        return log_prob
    
    
    def prob(self, component_log_prob, component_log_cdf, validate=True):
        """
        pdf of recalibrated linear pool ensemble
        
        Parameters
        ----------
        component_log_prob: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        validate: boolean
            Indicator of whether component_log_prob and component_log_cdf should
            be validated
        
        Returns
        -------
        Ensemble pdf value for all observation cases as a tensor of length N
        """
        return tf.math.exp(self.log_prob(component_log_prob, component_log_cdf,
                                         validate=validate))
    
        
    def log_cdf(self, component_log_cdf):
        """
        Log cdf of recalibrated linear pool ensemble
        
        Parameters
        ----------
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        
        Returns
        -------
        Log of ensemble cdf for all observation cases as a tensor of length N
        """
        # handle missing values; returns linear pool weights and updated
        # component model probabilities
        # note that we intentionally pass component_log_cdf as
        # component_log_prob to satisfy handle_missingness, then discard the
        # extra result. TODO: update handle_missingness so this isn't necessary
        _, component_log_cdf, lp_w = self.handle_missingness(
            component_log_prob=component_log_cdf,
            component_log_cdf=component_log_cdf,
            validate=False)
        
        # log_cdf for linear pool
        lp_log_cdf = tf.reduce_logsumexp(component_log_cdf + tf.math.log(lp_w),
                                          axis=1)
        
        # extract parameters for recalibration (all other than 'lp_w')
        rc_parameters = {k: v for k,v in self.parameters.items() if k != 'lp_w'}
        
        # log cdf for recalibrated linear pool
        # log[ G{ \sum_{m=1}^M \pi_m F_{m,i}(y); \theta } ]
        return self.rc_log_cdf(lp_log_cdf=lp_log_cdf, **rc_parameters)
    
    
    def cdf(self, component_log_cdf):
        """
        cdf of recalibrated linear pool ensemble
        
        Parameters
        ----------
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        
        Returns
        -------
        Ensemble cdf value for all observation cases as a tensor of length N
        """
        return tf.math.exp(self.log_cdf(component_log_cdf))
    
    
    def log_score_objective(self, component_log_prob, component_log_cdf, validate=True):
        """
        Log score objective function for use during parameter estimation
        
        Parameters
        ----------
        component_log_prob: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        validate: boolean
            Indicator of whether component_log_prob and component_log_cdf should
            be validated
        
        Returns
        -------
        Total log score over all predictions as scalar tensor
        """
        # negative sum of ensemble log pdf values across
        # observation indices i = 1, ..., N
        return -tf.reduce_sum(self.log_prob(component_log_prob,
                                            component_log_cdf,
                                            validate=validate))
    
    
    def fit(self,
            component_log_prob,
            component_log_cdf,
            optim_method = "adam",
            num_iter = 100,
            learning_rate = 0.1,
            verbose = False,
            save_frequency = None,
            save_path = None):
        """
        Estimate model parameters. This can be called multiple times; later
        calls will continue optimization from the results of earlier calls.
        
        Parameters
        ----------
        component_log_prob: 2D tensor with shape (N, M)
            Component log pdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        component_log_cdf: 2D tensor with shape (N, M)
            Component log cdf values for observation cases i = 1, ..., N,
            models m = 1, ..., M
        optim_method: string
            optional method for optimization.  Options are "adam" or "sgd".
        num_iter: integer
            Number of iterations for optimization
        learning_rate: Tensor or a floating point value.
            The learning rate
        verbose: boolean
            Indicator of whether logging messages should be printed
        save_frequency: integer or None
            Parameter estimates will be saved after every `save_frequency`
            optimization iterations
        save_path: string
            File path where parameter estimates will be saved
        
        Returns
        -------
        None
        """
        # convert inputs to float tensors
        component_log_prob = tf.convert_to_tensor(component_log_prob,
                                                  dtype=tf.float32)
        component_log_cdf = tf.convert_to_tensor(component_log_cdf,
                                                 dtype=tf.float32)
        
        # validate component_log_prob and component_log_cdf
        self.validate_component_log_prob_cdf(component_log_prob,
                                             component_log_cdf)
        
        # validate num_iter and save_frequency
        # validation of other arguments happens elsewhere
        if not isinstance(num_iter, int):
            raise ValueError('num_iter must be an int')
        
        if save_frequency == None:
            save_frequency = num_iter + 1
        
        if not isinstance(save_frequency, int):
            raise ValueError('save_frequency must be None or an int')
        
        # create optimizer
        if optim_method == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optim_method == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate = learning_rate)
        else:
            raise ValueError('optim_method must be either "adam" or "sgd"')
        
        # initiate loss trace
        lls_ = np.zeros(num_iter, np.float32)
        
        # list of trainable variables for which to calculate gradient
        trainable_variables = [self.parameters[v].trainable_variables[0] \
            for v in self.parameters.keys()]
        
        # apply gradient descent num_iter times
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                loss = self.log_score_objective(
                    component_log_prob=component_log_prob,
                    component_log_cdf=component_log_cdf,
                    validate=False)
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
                    'param_estimates': { k: v.numpy() \
                                         for k,v in self.parameters.items() },
                    'loss_trace': lls_
                }
                
                with open(str(save_path), 'wb') as outfile:
                    pickle.dump(params_to_save, outfile)
        
        # update loss trace
        self.loss_trace = np.concatenate([self.loss_trace, lls_])



class LinearPool(BaseRCLP):
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
        if not isinstance(M, int):
            raise ValueError('M must be an int')
        
        super(LinearPool, self).__init__(M=M, rc_parameters={})
    
    
    def rc_log_prob(self, lp_log_cdf):
        """
        Calculate the log density of a recalibrating transformation for an
        ensemble forecast. For a linear pool, no recalibration is done. This
        corresponds to the use of a Uniform(0, 1) distribution with log
        density that takes the value 0 everywhere.
        
        Parameters
        ----------
        lp_log_cdf: 1D tensor with length N
            log cdf values from the linear pool for observation cases i = 1, ..., N
        
        Returns
        -------
        1D tensor of length N
            log recalibration density evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N; all values are 0
        """
        return tf.zeros_like(lp_log_cdf)
    
    
    def rc_log_cdf(self, lp_log_cdf):
        """
        Calculate the log cdf of a recalibrating transformation for an
        ensemble forecast. For a linear pool, no recalibration is done. This
        corresponds to the use of a Uniform(0, 1) distribution with a cdf that
        is the identity function.
        
        Parameters
        ----------
        lp_log_cdf: 1D tensor with length N
            log cdf values from the linear pool for observation cases i = 1, ..., N
        
        Returns
        -------
        1D tensor of length N
            log recalibration cdf evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
        return lp_log_cdf



class BetaMixtureRCLP(BaseRCLP):
    def __init__(self, M, K) -> None:
        """
        Initialize a beta mixture re-calibrated linear pool model
        
        Parameters
        ----------
        M: integer
            number of component models
        K: integer
            number of beta mixture components
        
        Returns
        -------
        None
        """
        if not isinstance(M, int):
            raise ValueError('M must be an int')
        
        if not isinstance(K, int):
            raise ValueError('K must be an int')
        
        # Xavier initialization for pi
        if K == 1:
            pi_xavier_hw = tf.constant(1.0, dtype=tf.float32)
        else:
            pi_xavier_hw = 1.0 / tf.math.sqrt(tf.constant(K-1, dtype=tf.float32))
        
        softmax_bijector = tfb.SoftmaxCentered()
        def init_rc_pi():
            return softmax_bijector.forward(
                tf.random.uniform((K-1,), -pi_xavier_hw, pi_xavier_hw))
        
        softplus_bijector = tfb.Softplus()
        def init_rc_alpha_beta():
            return softplus_bijector.forward(
                tf.random.normal((K,)))
        
        rc_parameters = {
            'rc_pi': {
                'init': init_rc_pi,
                'bijector': softmax_bijector
            },
            'rc_alpha': {
                'init': init_rc_alpha_beta,
                'bijector': softplus_bijector
            },
            'rc_beta': {
                'init': init_rc_alpha_beta,
                'bijector': softplus_bijector
            }
        }
        
        super(BetaMixtureRCLP, self).__init__(M=M, rc_parameters=rc_parameters)
    
    
    def rc_log_prob(self, lp_log_cdf, rc_pi, rc_alpha, rc_beta):
        """
        Calculate the log density of a recalibrating transformation for an
        ensemble forecast. For a beta mixture recalibrated linear pool, the
        recalibrating transformation corresponds to a mixture of beta
        distributions, i.e., F(x) = \sum_{k=1}^K pi_k Beta(x | alpha_k, beta_k)
        
        Parameters
        ----------
        lp_log_cdf: 1D tensor with length N
            log cdf values from the linear pool for observation cases i = 1, ..., N
        rc_pi: tensor of length K
            mixture component weights
        rc_alpha: tensor of length K
            first shape parameter of Beta distribution for each mixture component
        rc_beta: tensor of length K
            second shape parameter of Beta distribution for each mixture component
        
        Returns
        -------
        1D tensor of length N
            log recalibration density evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
        # note that we transform lp_cdf away from 0 and 1 to avoid numerical
        # issues at the boundary of the support of the Beta distribution
        eps = 2.0 * np.finfo(np.float32).eps
        lp_cdf = tf.math.exp(lp_log_cdf) * (1.0 - 2.0 * eps) + eps
        return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=rc_pi),
                components_distribution=tfd.Beta(rc_alpha, rc_beta)
            ).log_prob(lp_cdf)
    
    
    def rc_log_cdf(self, lp_log_cdf, rc_pi, rc_alpha, rc_beta):
        """
        Calculate the log cdf of a recalibrating transformation for an
        ensemble forecast. For a beta mixture recalibrated linear pool, the
        recalibrating transformation corresponds to a mixture of beta
        distributions, i.e., F(x) = \sum_{k=1}^K pi_k Beta(x | alpha_k, beta_k)
        
        Parameters
        ----------
        lp_log_cdf: 1D tensor with length N
            log cdf values from the linear pool for observation cases i = 1, ..., N
        rc_pi: tensor of length K
            mixture component weights
        rc_alpha: tensor of length K
            first shape parameter of Beta distribution
        rc_beta: tensor of length K
            second shape parameter of Beta distribution
        
        Returns
        -------
        1D tensor of length N
            log recalibration cdf evaluated at the linear pool cdf values
            for each observation case i = 1, ..., N
        """
        # note that we transform lp_cdf away from 0 and 1 to avoid numeric issues
        # at the boundary of the support of the Beta distribution
        eps = np.finfo(np.float32).eps
        lp_cdf = tf.math.exp(lp_log_cdf) * (1.0 - 2.0 * eps) + eps
        return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=rc_pi),
                components_distribution=tfd.Beta(rc_alpha, rc_beta)
            ).log_cdf(lp_cdf)
