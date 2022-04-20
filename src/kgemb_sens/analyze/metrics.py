# -*- coding: utf-8 -*-

"""Methods for calculating metrics about network and test edge."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter
from scipy import stats
from scipy.stats import entropy
from kgemb_sens.transform.graph_utilities import edge_degree, undirect_multidigraph
from kgemb_sens.analyze.metrics_helpers import calc_scale_free_stats


# NOTE: Should do for "flat" and non-flat network? Or mostly just non-flat
def calc_edge_input_statistics(G, e, degree_dict, G_undir=None):
    print("Inside calc edge input stats...")
    if G_undir is None:
        G_undir = undirect_multidigraph(G)
    if len(e) > 2:
        if type(e[-1]) == dict:
            edge_rel = e[-1]['edge']
        else:
            edge_rel = e[-1]
    else:
        edge_rel = None
    u, v = e[:2]

    print("Creating a relation counter.")
    G_rel_set = [rel for _, _, rel in G.edges(data="edge")]
    # G = undirect_multidigraph(G)
    G_rel_counter = Counter(G_rel_set)
    print("Done with rel counter.")
    print(f"Edge rel: {edge_rel}")

    # Edge statistics
    edge_min_node_degree = min(degree_dict[u], degree_dict[v])
    edge_rel_count = G_rel_counter[edge_rel] - 1
    e_deg = edge_degree(G_undir, e, degree_dict)

    return edge_min_node_degree, edge_rel_count, e_deg


def calc_network_input_statistics(G, calc_expensive=False, G_undir=None):
    if G_undir is None:
        G_undir = undirect_multidigraph(G)
    G_rel_set = [rel for _, _, rel in G.edges(data="edge")]
    # G = undirect_multidigraph(G)
    G_rel_counter = Counter(G_rel_set)

    # Network statistics
    # n_ent_network = len(G.nodes())  # This fails because of singlet nodes.
    # Count nodes based on edges [triples]. Singlets aren't relevant to embedding pipeline.
    nodes = set(np.concatenate([(u, v) for u, v, _ in G.edges(data='edge')]))
    n_ent_network = len(nodes)
    n_rel_network = len(set(G_rel_set))
    n_triples = len(G.edges(data='edge'))
    n_conn_comps = nx.number_connected_components(G_undir)
    # n_triangles = sum(list(nx.triangles(G).values())) / 3
    med_rel_count = np.median(list(G_rel_counter.values()))
    min_rel_count = np.min(list(G_rel_counter.values()))

    # Entity and relational entropy statistics
    rel_entropy = entropy([G_rel_counter[rel] / n_triples for rel in G_rel_counter])
    G_sub_counter = Counter([s for s, _, _ in G.edges(data='edge')])
    G_obj_counter = Counter([o for _, o, _ in G.edges(data='edge')])
    ent_entropy = entropy([(G_sub_counter[ent] + G_obj_counter[ent]) / n_triples for ent in nodes])

    # return n_ent_network, n_rel_network, n_conn_comps, diam, avg_cc, n_triangles, med_rel_count, min_rel_count
    if calc_expensive:
        if n_conn_comps > 1:
            diam = float("inf")
        else:
            diam = nx.algorithms.distance_measures.diameter(G_undir)

        avg_cc = nx.average_clustering(nx.Graph(G_undir))  # to flatten
        return n_ent_network, n_rel_network, n_triples, n_conn_comps, med_rel_count, min_rel_count, rel_entropy, ent_entropy, avg_cc, diam

    else:
        return n_ent_network, n_rel_network, n_triples, n_conn_comps, med_rel_count, min_rel_count, rel_entropy, ent_entropy


def calc_powerlaw_statistics(degree_dict):
    lsf_stats, disc_mle_stats, powerlaw_stats = calc_scale_free_stats(degree_dict)
    lsf_b1, lsf_b2 = lsf_stats[0][0], lsf_stats[1][2]
    disc_mle_alpha_hat = disc_mle_stats[0]
    pl_alpha = powerlaw_stats[0]

    return lsf_b1, lsf_b2, disc_mle_alpha_hat, pl_alpha


def calc_output_statistics(prediction_ranks, degree_dict):
    entity_counts = [degree_dict[entity] for entity in prediction_ranks]
    entity_counts_pred_rank_spearman = stats.spearmanr(np.arange(len(prediction_ranks)), entity_counts).correlation
    return entity_counts_pred_rank_spearman
