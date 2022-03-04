# -*- coding: utf-8 -*-

"""Methods for calculating metrics about network and test edge."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter
from scipy import stats
from kgemb_sens.transform.graph_utilities import edge_degree, undirect_multidigraph
from kgemb_sens.analyze.metrics_helpers import calc_scale_free_stats


# NOTE: Should do for "flat" and non-flat network? Or mostly just non-flat
def calc_edge_input_statistics(G, e, degree_dict):
    edge_rel = e[-1]['edge']
    u, v = e

    G_rel_set = [rel for _, _, rel in G.edges(data="edge")]
    G = undirect_multidigraph(G)
    G_rel_counter = Counter(G_rel_set)

    # Edge statistics
    edge_min_node_degree = min(degree_dict[u], degree_dict[v])
    edge_rel_count = G_rel_counter[edge_rel]- 1
    e_deg = edge_degree(G, e, degree_dict)

    return edge_min_node_degree, edge_rel_count, e_deg


def calc_network_input_statistics(G, degree_dict):
    G_rel_set = [rel for _, _, rel in G.edges(data="edge")]
    G = undirect_multidigraph(G)
    G_rel_counter = Counter(G_rel_set)

    # Network statistics
    n_ent_network = len(G.nodes())
    n_rel_network = len(set(G_rel_set))
    diam = nx.algorithms.distance_measures.diameter(G)
    avg_cc = nx.average_clustering(G)
    n_triangles = sum(list(nx.triangles(G).values())) / 3
    med_rel_count = np.median(list(G_rel_counter.values()))
    min_rel_count = np.min(list(G_rel_counter.values()))

    return n_ent_network, n_rel_network, diam, avg_cc, n_triangles, med_rel_count, min_rel_count


def calc_powerlaw_stats(G, degree_dict):
    lsf_stats, disc_mle_stats, powerlaw_stats = calc_scale_free_stats(degree_dict)
    lsf_b1, lsf_b2 = lsf_stats[0][0], lsf_stats[1][2]
    disc_mle_alpha_hat = disc_mle_stats[0]
    pl_alpha = powerlaw_stats[0]

    return lsf_b1, lsf_b2, disc_mle_alpha_hat, pl_alpha


def calc_output_statistics(G, prediction_ranks, degree_dict):
    entity_counts = [degree_dict[entity] for entity in prediction_ranks]
    entity_counts_pred_rank_spearman = stats.spearmanr(np.arange(len(prediction_ranks)), entity_counts)
    return entity_counts_pred_rank_spearman
