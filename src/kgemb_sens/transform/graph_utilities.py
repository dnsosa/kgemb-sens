# -*- coding: utf-8 -*-

"""Utilities (helper functions) for KG processing pipeline, which entails manipulating KGs."""
import random
import networkx as nx
import numpy as np


def undirect_multidigraph(G):
    # NOTE: This function is needed because of weird behavior of the .to_undirected() method in Networkx. e.g. if I have
    # a network with (0, 1, "r1"), (0, 1, "r2"), (1, 0, "r1"); to_undirected() would convert this to two edges, not
    # three. Check documentation, it's bad: https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.to_undirected.html#networkx.MultiDiGraph.to_undirected
    G_undir = nx.MultiGraph()
    G_undir.add_edges_from(G.edges(data=True))
    return G_undir


def edge_dist(e1, e2, dist_mat):
    # TODO: Check that the edge is in the graph
    n11, n12, n21, n22 = e1[0], e1[1], e2[0], e2[1]

    if ((n11, n12) == (n21, n22)) or ((n11, n12) == (n22, n21)):
        # Connected between same pair of nodes but different relation types,
        if ((len(e1) >= 3) and (len(e2) >= 3)) and (e1[2] != e2[2]):
            return 1
        else:
            return 0

    # Handle the case if they're in different components
    if n22 not in dist_mat[n11]:
        return float("inf")

    d1121, d1122, d1221, d1222 = dist_mat[n11][n21] + 1, dist_mat[n11][n22] + 1, dist_mat[n12][n21] + 1, dist_mat[n12][
        n22] + 1
    return min(d1121, d1122, d1221, d1222)


def edge_degree(G, e, degree_dict):
    # TODO: Check that the edge is in the graph
    # NOTE: Ignores directionality
    # NOTE: Assumes the edge is in the graph

    print(degree_dict)
    print(G.to_undirected().number_of_edges(e[0], e[1]))
    print(degree_dict[e[0]])
    print(degree_dict[e[1]])
    return (degree_dict[e[0]] - 1) + (degree_dict[e[1]] - undirect_multidigraph(G).number_of_edges(e[0], e[1]))


def prob_dist(edge,
              all_edges,
              dist_mat=None,
              degree_dict=None,
              prob_type="distance",
              graph=None,
              alpha=0):
    if prob_type == "distance":
        if dist_mat is None:
            print("No distance matrix provided!")
        u_dist = np.array([edge_dist(edge, other_edge, dist_mat) for other_edge in all_edges])
        inf_mask = (u_dist != np.inf)
        # Proximity parameter: positive = prefer far, 0 = uniform, negative = prefer near
        u_dist = u_dist ** (float(alpha))
        # Deal with 0 distances
        u_dist[u_dist == np.inf] = 0
        # Deal with 0 distance in the case of alpha = 0. Probably redundant with above
        if edge in all_edges:  # might not be the case in the contradictification pipeline
            edge_idx = all_edges.index(edge)
            u_dist[edge_idx] = 0
        # Deal with making the distances to the different components (infinite) equal to 0
        u_dist *= inf_mask

    elif prob_type == "degree":
        if degree_dict is None:
            print("No distance matrix provided!")
        u_dist = np.array([edge_degree(graph, other_edge, degree_dict) for other_edge in all_edges])
        # Degree priority parameter: positive = prefer hubs, 0 = uniform, negative = deprioritize hubs
        u_dist = u_dist ** (float(alpha))
        # Deal with 0 distances
        u_dist[u_dist == np.inf] = 0
        # Deal with 0 distance in the case of alpha = 0. Probably redundant with above
        if edge in all_edges:
            edge_idx = all_edges.index(edge)
            u_dist[edge_idx] = 0

    # Replace NaNs with 0s
    u_dist = np.nan_to_num(u_dist)
    # Normalize
    p_dist = u_dist / sum(u_dist)

    return p_dist


def prob_dist_from_list(edge_set,
                        all_edges,
                        dist_mat=None,
                        degree_dict=None,
                        prob_type="distance",
                        graph=None,
                        alpha=0):
    u_dist = np.zeros(len(all_edges))
    # Calculate probability dists based on individual edges, then sum together
    for edge in edge_set:
        u_dist += prob_dist(edge,
                            all_edges,
                            dist_mat,
                            degree_dict,
                            prob_type,
                            graph,
                            alpha)

    # Make sure probability of test edges is 0
    for edge in edge_set:
        if edge in all_edges:  # NEW CHANGE
            edge_idx = all_edges.index(edge)
            u_dist[edge_idx] = 0

    p_dist = u_dist / sum(u_dist)

    return p_dist


def random_split_list(in_list, val_frac):
    random.shuffle(in_list)
    k = int(np.round(len(in_list) * val_frac))
    return in_list[:k], in_list[k:]


