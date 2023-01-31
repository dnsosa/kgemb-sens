"""General utilities for all code."""

# -*- coding: utf-8 -*-

import math

import networkx as nx


def good_round(x):
    frac = x - math.floor(x)
    if frac < 0.5:
        return math.floor(x)
    return math.ceil(x)


def net2df(G):
    return nx.to_pandas_edgelist(G, edge_key="key")