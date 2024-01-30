from functools import wraps
import isuelogit
import time
import os
from typing import List
import tensorflow as tf
import pesuelogit.networks
from isuelogit.paths import Path
import isuelogit.printer as printer
import csv
import numpy as np

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper

def read_paths(block_output=True, **kwargs):
    network = kwargs['network']
    if block_output:
        with isuelogit.printer.block_output(show_stdout=False, show_stderr=False):
            pesuelogit.networks.read_paths(**kwargs)
        print(f'{len(network.paths)} paths were read and incidence matrices were built')
    else:
        pesuelogit.networks.read_paths(**kwargs)

def load_k_shortest_paths(block_output=True, theta=None, **kwargs):
    network = kwargs['network']
    # if theta is None:
    #     #Use free flow travel time to compute shortest path
    #     kwargs['theta'] = {'tt':-1}
    if block_output:
        with isuelogit.printer.block_output(show_stdout=False, show_stderr=False):
            pesuelogit.networks.load_k_shortest_paths(**kwargs)
        print(f'{len(network.paths)} paths were loaded and incidence matrices were built')
    else:
        pesuelogit.networks.load_k_shortest_paths(**kwargs)

def write_paths(paths: List[Path], filepath=None):
    t0 = time.time()
    lines = []
    total_paths = len(paths)
    for path, counter in zip(paths, range(total_paths)):
        # printer.printProgressBar(counter, total_paths-1, prefix='Writing paths:', suffix='', length=20)
        line = []
        for node in path.nodes:
            line.append(node.key)
        lines.append(line)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(lines)
    print(str(total_paths) + ' paths were written in ' + str(np.round(time.time() - t0, 1)) + '[s]')

def flat_od_from_generated_trips(generated_trips, ods):
    o, d = np.array(ods).T
    counts_array = np.tile(o, (generated_trips.shape[0], 1)).copy()
    # Count the number of unique values and create a new array
    for i, row in enumerate(counts_array):
        # Get the unique values and their counts
        unique_values, counts = np.unique(row, return_counts=True)
        # Fill the new array with the counts of unique values
        unique_counts = counts[np.searchsorted(unique_values, row)]
        counts_array[i] = unique_counts.flatten()
    return tf.experimental.numpy.take(generated_trips, o, axis=1).numpy() / counts_array
