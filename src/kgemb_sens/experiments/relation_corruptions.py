# -*- coding: utf-8 -*-

"""Perturb the input KG relations to see how relation quality will affect performance."""

import networkx as nx

import numpy as np

import pandas as pd


def corrupt_rels_list(in_rels_list, corrupt_frac, whitelist_rels, all_possible_rels, SEED):
    """
    Corrupt a fraction of relations from a list.

    :param in_rels_list: input list of relations
    :param corrupt_frac: fraction of relations to corrupt
    :param whitelist_rels: list of relations to NOT corrupt
    :param all_possible_rels: all possible relations to sample from in the corruption process
    :param SEED: random seed
    """
    np.random.seed(SEED)

    out_list = in_rels_list.copy()
    for i, rel in enumerate(in_rels_list):
        # Ensure relation not in whitelist--if it isn't, then corrupt it with corrupt_frac probability
        if (rel not in whitelist_rels) and (np.random.sample(1) <= corrupt_frac):
            out_list[i] = np.random.choice(all_possible_rels)

    return out_list


def corrupt_relations(G, corrupt_frac, whitelist_rels, all_possible_rels, SEED=42):
    """
    Corrupt a fraction of relations in the KG.

    :param G: input KG
    :param corrupt_frac: fraction of relations to corrupt
    :param whitelist_rels: list of relations to NOT corrupt
    :param all_possible_rels: all possible relations to sample from in the corruption process
    :param SEED: random seed
    """
    G_df = nx.to_pandas_edgelist(G, edge_key="key")
    rels_list = G_df["rel"]
    corrupted_rels_list = corrupt_rels_list(rels_list, corrupt_frac, whitelist_rels, all_possible_rels, SEED)
    G_df["rel"] = corrupted_rels_list
    G_corrupt = nx.from_pandas_edgelist(G_df, edge_key="key", edge_attr="rel", create_using=nx.MultiDiGraph())
    return G_corrupt
