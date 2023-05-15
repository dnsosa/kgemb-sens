# -*- coding: utf-8 -*-

"""Perturb the input KG to see how topology will affect performance."""

import networkx as nx
import numpy as np

from collections import Counter
from scipy.stats import entropy

from ..utilities import net2df
from ..transform.graph_utilities import avg_node_degree, edge_degree, entity_entropy


# TODO: Implement this


def add_self_loops(G, fill_frac, SEED, upper_bound=500):
    """
    Add self-loops to the graph to artificially but not meaningfully flatten ESP distribution.

    :param G: input KG
    :param fill_frac: fraction of the self-loops needed to make an equal degree KG to include
    :param SEED: random seed
    :param upper_bound: degree upper bound to inflate a node
    """
    np.random.seed(SEED)

    G_df = net2df(G)
    deg_dict = dict((Counter(G_df.source) + Counter(G_df.target)))
    max_degree = min(max(deg_dict.values()), upper_bound)

    self_loops_to_add = []
    for node, degree in deg_dict.items():
        if degree < max_degree:
            short_of_max = max_degree - degree
            self_edge = (node, node, {"edge": "is"})
            self_loops_to_add += ([self_edge] * short_of_max)

    idxs = np.random.choice(np.arange(len(self_loops_to_add)), size=round(fill_frac * len(self_loops_to_add)))
    self_loops_sample = [self_loops_to_add[idx] for idx in idxs]

    # Convert back so NX can deal with the key ids
    G_prime = nx.from_pandas_edgelist(G_df, edge_key="key", edge_attr="edge", create_using=nx.MultiDiGraph())
    G_prime.add_edges_from(self_loops_sample)

    return G_prime


def remove_hubs(G, frac_nodes, n_hubs=0, SEED=42):
    """
    Remove hub nodes from the graph to artificially AND meaningfully flatten out ESP distribution.

    :param G: input KG
    :param frac_nodes: fraction of nodes to remove (remove in degree order)
    :param n_hubs: number of hubs to remove
    :param SEED: random seed
    """
    pass # TODO: See implementation in transformation.graph_utilities


def upsample_low_deg_triples(G,
                             alpha=-2,
                             max_num_attempts=100000,
                             batch_size=5000,
                             multiplier=4,
                             SEED=42):

    np.random.seed(SEED)

    G_df = net2df(G)

    n_attempts = 0

    # emd = edge max degree
    og_node_deg_dict = dict((Counter(G_df.source) + Counter(G_df.target)))
    all_edges = list(G_df.itertuples(index=False, name=None))
    edge_degs = np.array([edge_degree(G, edge, og_node_deg_dict) for edge in all_edges])
    og_edge_deg_dict = dict(zip(all_edges, edge_degs))

    avg_node_degs = np.array([avg_node_degree(edge, og_node_deg_dict) for edge in all_edges])
    og_avg_node_degs_dict = dict(zip(all_edges, avg_node_degs))

    max_entropy = entropy(np.ones(G.number_of_nodes()) / G.number_of_nodes())

    # Will need to be updating these two dicts as we add to network
    edge_deg_dict = og_edge_deg_dict
    node_deg_dict = og_node_deg_dict
    avg_node_degs_dict = og_avg_node_degs_dict

    n_triples = len(G_df)

    while (n_triples <= multiplier * len(G_df)) and (n_attempts < max_num_attempts):

        # Calculate the probability distribution
        u_dist = (np.array(list(avg_node_degs_dict.values())) + 1) ** float(alpha)
        p_dist = u_dist / np.sum(u_dist)
        sampled_triples = np.random.multinomial(batch_size, p_dist, size=1).flatten()

        # Update edge degree dict
        edge_deg_dict = dict(zip(list(edge_deg_dict.keys()), np.array(list(edge_deg_dict.values())) + sampled_triples))

        # Update node degree dict
        edge_list = list(edge_deg_dict.keys())
        for idx, count in enumerate(sampled_triples):
            if sampled_triples[idx] == 0:
                continue
            u, v, k, r = edge_list[idx]
            node_deg_dict[u] += count  # if downsample, then subtract
            node_deg_dict[v] += count

        # Update avg_node_degs_dict
        avg_node_degs = np.array([avg_node_degree(edge, node_deg_dict) for edge in all_edges])
        avg_node_degs_dict = dict(zip(all_edges, avg_node_degs))

        # Calculate the new ent entropy
        n_triples = len(G_df) + batch_size * n_attempts
        ent_entropy = entity_entropy(node_deg_dict, n_triples)
        print(f"After iter: {n_attempts}, entity entropy = {ent_entropy}, frac entropy = {ent_entropy / max_entropy}")

        n_attempts += 1

    # Create the list of new triples
    difference_edge_degs = np.array(list(edge_deg_dict.values())) - np.array(list(og_edge_deg_dict.values()))
    difference_edge_deg_dict = dict(zip(edge_deg_dict.keys(), difference_edge_degs))

    sampled_triples_list = []
    for (u, v, k, r), edge_deg in difference_edge_deg_dict.items():
        edge_formatted = (u, v, {"edge": r})
        sampled_triples_list += ([edge_formatted] * edge_deg)

    # Finally update the graph
    G_prime = nx.from_pandas_edgelist(G_df, edge_key="key", edge_attr="edge", create_using=nx.MultiDiGraph())
    G_prime.add_edges_from(sampled_triples_list)

    return G_prime


def downsample_high_deg_triples(G,
                                alpha=2,
                                SEED=42,
                                batch_size=5000,
                                remaining_fraction=0.25):
    np.random.seed(SEED)

    G_df = net2df(G)

    n_attempts = 0

    og_node_deg_dict = dict((Counter(G_df.source) + Counter(G_df.target)))
    all_edges = list(G_df.itertuples(index=False, name=None))

    avg_node_degs = np.array([avg_node_degree(edge, og_node_deg_dict) for edge in all_edges])
    og_avg_node_degs_dict = dict(zip(all_edges, avg_node_degs))

    ent_entropy = entity_entropy(og_node_deg_dict, len(all_edges) + 100 * n_attempts)
    max_entropy = entropy(np.ones(G.number_of_nodes()) / G.number_of_nodes())

    print(ent_entropy)

    # Will need to be updating these two dicts as we remove from the network
    node_deg_dict = og_node_deg_dict
    avg_node_degs_dict = og_avg_node_degs_dict

    n_triples = len(G_df)

    # Will need to maintain the current edges left
    edge_list = all_edges

    while n_triples >= remaining_fraction * len(G_df):

        # Calculate the probability distribution
        u_dist = (np.array(list(avg_node_degs_dict.values())) + 1) ** float(alpha)
        p_dist = u_dist / np.sum(u_dist)
        sampled_triples = np.random.choice(len(p_dist), batch_size, replace=False, p=p_dist)
        sampled_triples_edges = [edge_list[idx] for idx in sampled_triples]

        # Update edge degree dict
        edge_list = list(set(edge_list).difference(set(sampled_triples_edges)))

        # Update node degree dict
        for edge in sampled_triples_edges:
            u, v, k, r = edge
            node_deg_dict[u] -= 1
            node_deg_dict[v] -= 1

        # Update avg_node_degs_dict
        avg_node_degs = np.array([avg_node_degree(edge, node_deg_dict) for edge in edge_list])
        avg_node_degs_dict = dict(zip(edge_list, avg_node_degs))

        # Calculate the new ent entropy
        n_triples = len(G_df) - batch_size * n_attempts
        ent_entropy = entity_entropy(node_deg_dict, n_triples)
        print(f"After iter: {n_attempts}, entity entropy = {ent_entropy}, frac entropy = {ent_entropy / max_entropy}")
        print(f"{n_triples} remaining...")

        n_attempts += 1

    G_prime = nx.MultiDiGraph([(u, v, {"edge": r}) for (u, v, _, r) in edge_list])
    return G_prime
