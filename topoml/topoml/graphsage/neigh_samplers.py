from __future__ import division
from __future__ import print_function

from .layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, name='neighbor_sampler', **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.name = name

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids, name = self.name)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)), name = self.name)
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples], name=self.name)
        return adj_lists
