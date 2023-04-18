# -*- coding: utf-8 -*-

"""Perturb the input KG to see how topology will affect performance."""

import networkx as nx
import numpy as np

from collections import Counter

from ..utilities import net2df


# TODO: Implement this


def add_self_loops(G, fill_frac, SEED):
    """
    Add self-loops to the graph to artificially but not meaningfully flatten ESP distribution.

    :param G: input KG
    :param fill_frac: fraction of the self-loops needed to make an equal degree KG to include
    :param SEED: random seed
    """
    np.random.seed(SEED)

    G_df = net2df(G)
    deg_dict = dict((Counter(G_df.source) + Counter(G_df.target)))
    max_degree = max(deg_dict.values())

    self_loops_to_add = []
    for node, degree in deg_dict.items():
        if degree < max_degree:
            short_of_max = max_degree - degree
            self_edge = (node, node, {"rel": "is"})
            self_loops_to_add += ([self_edge] * short_of_max)

    idxs = np.random.choice(np.arange(len(self_loops_to_add)), size=round(fill_frac * len(self_loops_to_add)))
    self_loops_sample = [self_loops_to_add[idx] for idx in idxs]

    # Convert back so NX can deal with the key ids
    G_prime = nx.from_pandas_edgelist(G_df, edge_key="key", edge_attr="rel", create_using=nx.MultiDiGraph())
    G_prime.add_edges_from(self_loops_sample)


def remove_hubs(G, frac_nodes, n_hubs=0, SEED=42):
    """
    Remove hub nodes from the graph to artificially AND meaningfully flatten out ESP distribution.

    :param G: input KG
    :param frac_nodes: fraction of nodes to remove (remove in degree order)
    :param n_hubs: number of hubs to remove
    :param SEED: random seed
    """
    pass


def upsample_low_deg_triples(G, frac_triples, SEED=42):
    """
    Upsample low-degree triples by creating multi-edges to artificially but not meaningfully flatten out ESP dist.

    :param G: input KG
    :param frac_triples: fraction of relations to upsample
    :param SEED: random seed
    """
    pass


def degree_based_downsample(G, frac_triples, SEED=42):
    """
    Downsample input KG based on degree to artificially AND meaningfully flatten out ESP distribution.

    :param G: input KG
    :param frac_triples: fraction of relations to downsample
    :param SEED: random seed
    """
    pass
