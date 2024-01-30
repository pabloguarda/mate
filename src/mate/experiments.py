import isuelogit as isl
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple
from mate.models import mate
from mate.metrics import zscore, nmse
from isuelogit.printer import block_output

def simulate_features(n_days, time_variation=False, **kwargs) -> pd.DataFrame:
    """
    
    :param n_days: 
    :param time_variation: when True, the values of the exogenous features change between different timepoints 
    :param kwargs: 
    :return: 
    """
    linkdata_generator = isl.factory.LinkDataGenerator()
    df_list = []
    for i in range(1, n_days + 1):
        if i == 1 or time_variation:
            df_day = linkdata_generator.simulate_features(**kwargs)
            df_day.insert(0, 'timepoint', i)
        else:
            df_day = df_day.assign(timepoint=i)
        df_list.append(df_day)
    df = pd.concat(df_list)
    return df

def simulate_mtp_data(model: mate,
                             X,
                             optimizer,
                             threshold_relative_gap=1e-5,
                             max_epochs=100,
                             loss_metric=None,
                             batch_size=None,
                             # coverage = 1,
                             sd_x: float = 0,
                             sd_t: float = 0):
    """
    :param model:
    :param X: The last column has the period id. The remaining columns correspond to the exogenous features
    :param learning_rate:
    :param threshold_relative_gap:
    :param max_epochs:
    :param coverage:
    :param sd_x:  the standard deviation relative to the true mean of traffic counts
    :param sd_t:
    :return:
    """
    if loss_metric is None:
        loss_metric = zscore
    model.compute_equilibrium(tf.cast(X, dtype=model.dtype),
                              loss_metric=loss_metric,
                              optimizer=optimizer,
                              batch_size=batch_size,
                              loss_weights={'equilibrium': 1},
                              threshold_relative_gap=threshold_relative_gap,
                              # epochs_print_interval= _EPOCHS_PRINT_INTERVAL,
                              epochs=max_epochs)
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))
    # This should end in a single epoch as the relative threshold is the same than for computing equlibrium
    with block_output(show_stdout=False, show_stderr=False):
        Y_pred = model.predict(tf.cast(X, dtype=model.dtype),
                               period_dict={k: v for k, v in model.period_dict.items()},
                               pretrain_link_flows=False,
                               loss_metric=loss_metric,
                               optimizer=optimizer,
                               batch_size=batch_size,
                               loss_weights={'equilibrium': 1},
                               threshold_relative_gap=threshold_relative_gap,
                               epochs=1)
    traveltimes, link_flows = tf.unstack(Y_pred, axis=-1)
    linkdata_generator = isl.factory.LinkDataGenerator()
    noisy_flow = linkdata_generator.add_error_counts(original_counts=link_flows.numpy(), sd_x=sd_x)
    noisy_traveltime = linkdata_generator.add_error_counts(original_counts=traveltimes.numpy(), sd_x=sd_t)
    return tf.stack([noisy_traveltime, noisy_flow], axis=2)