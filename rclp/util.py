import numpy as np
import tensorflow as tf

def handle_missingness(component_log_prob, component_log_cdf, w):
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
    w: 1D tensor with length M
        Component model weights, where `w[m]` is the weight given to model m

    Returns
    -------
    component_log_prob_nans_replaced: 2D tensor with shape (N, M)
        Component log-pdf values with nans replaced with 0
    component_log_cdf_nans_replaced: 2D tensor with shape (N, M)
        Component log-cdf values with nans replaced with 0
    broadcast_w: 2D tensor with shape (N, M)
        broadcast_w has N copies of the argument w with weights w[i,m] set
        to 0 at indices where log_f[i,m] is nan. The weights are then
        re-normalized to sum to 1 within each combination of i.
    """
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
                                                np.float32(0.0),
                                                np.float32(component_log_prob))
    component_log_cdf_nans_replaced = tf.where(missing_mask,
                                               np.float32(0.0),
                                               np.float32(component_log_cdf))
    
    return component_log_prob_nans_replaced, component_log_cdf_nans_replaced, broadcast_w
