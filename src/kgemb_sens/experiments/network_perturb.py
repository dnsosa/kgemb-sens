# -*- coding: utf-8 -*-

"""Perturb the input KG to see how topology will affect performance."""

import pandas as pd
import networkx as nx


def add_self_loops(G):
    """
    Add self-loops to the graph to artificially but not meaningfully flatten ESP distribution.

    :param G: input KG
    """
    pass


def remove_hubs(G):
    """
    Remove hub nodes from the graph to artificially AND meaningfully flatten out ESP distribution.

    :param G: input KG
    """
    pass


def upsample_low_deg_triples(G):
    """
    Upsample low-degree triples by creating multi-edges to artificially but not meaningfully flatten out ESP dist.

    :param G: input KG
    """
    pass


def degree_based_downsample(G, SEED=42):
    """
    Downsample input KG based on degree to artificially AND meaningfully flatten out ESP distribution.

    :param G: input KG
    :param SEED: random seed
    """
    pass
