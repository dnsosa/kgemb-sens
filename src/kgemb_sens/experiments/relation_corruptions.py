# -*- coding: utf-8 -*-

"""Perturb the input KG relations to see how relation quality will affect performance."""

import networkx as nx

import numpy as np

from ..utilities import net2df


def corrupt_rels_list(in_rels_list, corrupt_frac, whitelist_rels, all_possible_rels, SEED):
    """
    Corrupt a fraction of relations from a list.

    :param in_rels_list: input list of relations
    :param corrupt_frac: fraction of relations to corrupt
    :param whitelist_rels: list of relations to NOT corrupt
    :param all_possible_rels: all possible relations to sample from in the corruption process
    :param SEED: random seed
    """
    # TODO: Emulate randomize_edges condition in graph_utilities?
    np.random.seed(SEED)

    out_list = in_rels_list.copy()
    for i, rel in enumerate(in_rels_list):
        # Ensure relation not in whitelist--if it isn't, then corrupt it with corrupt_frac probability
        if (rel not in whitelist_rels) and (np.random.sample(1) <= corrupt_frac):
            out_list[i] = np.random.choice(all_possible_rels)

    return out_list


def flatten_rels_list(in_rels_list, flatten_frac, whitelist_rels, flat_rel, SEED):
    """
    Flatten a fraction of relations from a list to a single relation type.

    :param in_rels_list: input list of relations
    :param flatten_frac: fraction of relations to flatten
    :param whitelist_rels: list of relations to NOT flatten
    :param flat_rel: the name of the single relation to flatten to
    :param SEED: random seed
    """
    np.random.seed(SEED)

    out_list = in_rels_list.copy()
    for i, rel in enumerate(in_rels_list):
        # Ensure relation not in whitelist--if it isn't then flatten it with flatten_frac probability
        if (rel not in whitelist_rels) and (np.random.sample(1) <= flatten_frac):
            out_list[i] = flat_rel

    return out_list


def perturb_relations(G, perturb_frac, whitelist_rels, all_possible_rels, flat_rel="blah", perturb_method="corrupt",
                      SEED=42):
    """
    Corrupt a fraction of relations in the KG.

    :param G: input KG
    :param perturb_frac: fraction of relations to perturb
    :param whitelist_rels: list of relations to NOT perturb
    :param all_possible_rels: all possible relations to sample from in the perturbation process (if corrupting)
    :param flat_rel: the name of the single relation to flatten to (if flattening)
    :param perturb_method: how to perturb relations--flatten or corrupt
    :param SEED: random seed
    """
    # Get list of relations
    G_df = net2df(G)
    rels_list = G_df["edge"]

    # Perform perturbation on list of relations
    if perturb_method == "flatten":
        perturbed_rels_list = flatten_rels_list(rels_list, perturb_frac, whitelist_rels, flat_rel, SEED)
    elif perturb_method == "corrupt":
        perturbed_rels_list = corrupt_rels_list(rels_list, perturb_frac, whitelist_rels, all_possible_rels, SEED)

    # Update the list of relations with the new perturbed list
    G_df["edge"] = perturbed_rels_list
    G_corrupt = nx.from_pandas_edgelist(G_df, edge_key="key", edge_attr="edge", create_using=nx.MultiDiGraph())
    return G_corrupt
