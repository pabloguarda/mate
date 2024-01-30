import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics

def error(actual: tf.Tensor, predicted: tf.Tensor, mask=None):
    # return tf.boolean_mask(predicted - actual, tf.math.is_finite(predicted - actual))
    if mask is None:
        mask = tf.math.is_finite(predicted - actual)
    return tf.boolean_mask(predicted - actual, mask)

def l1norm(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.norm(error(actual, predicted), 1)

def sse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.reduce_sum(tf.math.pow(error(actual, predicted), 2))

def mse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.reduce_mean(tf.math.pow(error(actual, predicted), 2))

def mape(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """
    # Skip cases where the observed values are equal to zero or nan to avoid getting an undefined mape
    :param actual:
    :param predicted:
    :param weight:
    :return:
    """
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    actual, predicted = [tf.cast(tf.constant(i), tf.float32) if not isinstance(i, tf.Tensor) else tf.cast(i, tf.float32)
                         for i in [actual, predicted]]
    mask = tf.cast(tf.math.is_finite(actual), tf.int32) * tf.cast(actual > 0, tf.int32)
    # mask = tf.cast(tf.math.is_finite(actual), tf.int32)
    return 100 * weight * tf.reduce_mean(tf.abs(error(actual, predicted, mask=mask)) / tf.boolean_mask(actual, mask))

def mdape(actual: tf.Tensor, predicted: tf.Tensor):
    ''' Median absolute percentage error:
    Skip cases where the observed values are equal to zero or nan to avoid getting an undefined mape
    '''
    actual, predicted = [tf.cast(tf.constant(i), tf.float32) if not isinstance(i, tf.Tensor) else tf.cast(i, tf.float32)
                         for i in [actual, predicted]]
    mask = tf.cast(tf.math.is_finite(actual), tf.int32) * tf.cast(actual > 0, tf.int32)
    # y_true, y_pred = np.array(actual), np.array(predicted)
    return 100 * np.median(tf.abs(error(actual, predicted, mask=mask)) / tf.boolean_mask(actual, mask))

def median_percentage_error(actual: tf.Tensor, predicted: tf.Tensor):
    mask = tf.cast(tf.math.is_finite(actual), tf.int32) * tf.cast(actual > 0, tf.int32)
    # y_true, y_pred = np.array(actual), np.array(predicted)
    return np.median(error(actual, predicted, mask=mask) / tf.boolean_mask(actual, mask)) * 100

def r2_score(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    mask = tf.math.is_finite(predicted - actual)
    # if predicted.shape[0] < actual.shape[0]:
    #     predicted = tf.repeat(predicted, actual.shape[0], axis = 0)
    return weight * metrics.r2_score(y_true=tf.boolean_mask(actual, mask),
                                     y_pred=tf.boolean_mask(predicted, mask))

def rmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.math.sqrt(mse(actual, predicted))

def nmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # return weight*rmse(actual, predicted)/tf.experimental.numpy.nanmean(actual)
    if np.isnan(actual).all():
        return weight * mse(actual, predicted)
    return weight * mse(actual, predicted) / tf.experimental.numpy.nanmean(actual) ** 2

def nrmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # return weight*rmse(actual, predicted)/tf.experimental.numpy.nanmean(actual)
    if np.isnan(actual).all():
        return weight * rmse(actual, predicted)
    return weight * rmse(actual, predicted) / tf.experimental.numpy.nanmean(actual)

def zscore(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # TODO: Mask standard deviation calculation
    # if np.isnan(actual).all():
    #     return weight*mse(actual, predicted)
    #
    # return weight*mse(actual, predicted)/np.nanstd(actual)**2
    if np.isnan(actual).all():
        return weight * rmse(actual, predicted)
    return weight * rmse(actual, predicted) / np.nanstd(actual)

def z2score(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # TODO: Mask standard deviation calculation
    if np.isnan(actual).all():
        return weight * mse(actual, predicted)
    return weight * mse(actual, predicted) / np.nanstd(actual) ** 2

def mnrmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """ Normalized rmse by the maximum observed value"""
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * rmse(actual, predicted) / tf.experimental.numpy.max(actual[~tf.experimental.numpy.isnan(actual)])

def btcg_mse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    ''' Normalization used by Wu et al. (2018), TRC. This metric has more numerical issues than using MSE'''
    rel_error = tf.math.divide_no_nan(predicted, actual)
    return 1 / 2 * tf.reduce_mean(tf.math.pow(tf.boolean_mask(rel_error, tf.math.is_finite(rel_error)) - 1, 2))
    # return 1 / 2 * tf.reduce_mean(tf.math.pow(error(actual, predicted) /
    #                                           (tf.boolean_mask(actual, tf.math.is_finite(actual)) + epsilon), 2))
