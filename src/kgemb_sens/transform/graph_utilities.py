# -*- coding: utf-8 -*-

"""Utilities (helper functions) for KG processing pipeline, which entails manipulating KGs."""
import random
import networkx as nx
import numpy as np

from kgemb_sens.utilities import good_round


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

    # Handle the case if they're in different components
    if n22 not in dist_mat[n11]:
        return float("inf")

    # Handle the 2 and 3-tuple (undir net vs KG) cases differently
    # If the nodes in the two edges are the same, then they're identical edges in a simple undirected net
    if (len(e1) == 2) and (len(e2) == 2):
        if ((n11, n12) == (n21, n22)) or (n11, n12) == (n22, n21):
            return 0
    # If the two edges are exactly the same (same rel type), then can be equal to 0
    elif (len(e1) >= 3) and (len(e2) >= 3):
        if e1 == e2:
            return 0
    else:
        assert "Edges of different lengths trying to be compared!"
        return None

    d1121, d1122, d1221, d1222 = dist_mat[n11][n21], dist_mat[n11][n22], dist_mat[n12][n21], dist_mat[n12][n22]
    return min(d1121, d1122, d1221, d1222) + 1


def edge_degree(G_undir, e, degree_dict):
    # TODO: Check that the edge is in the graph
    # NOTE: Ignores directionality
    # NOTE: Assumes the edge is in the graph
    ## G_undir = undirect_multidigraph(G)
    return (degree_dict[e[0]] - 1) + (degree_dict[e[1]] - G_undir.number_of_edges(e[0], e[1]))


def min_node_degree(e, degree_dict):
    u, v = e[0], e[1]
    return min(degree_dict[u], degree_dict[v])


def prob_dist(edge,  # This edge is the input edge we want to calculate a distance away from, for example
              all_edges,
              dist_mat=None,
              degree_dict=None,
              prob_type="distance",
              graph=None,
              alpha=0,
              min_edeg=0,
              max_edeg=float("inf"),
              min_mnd=0,
              max_mnd=float("inf"),
              rel_whitelist=None,
              whitelist_pairs=None):
    # NOTE: Assumes that graph is the undirected version
    e_degs = np.array([edge_degree(graph, other_edge, degree_dict) for other_edge in all_edges])
    e_mnds = np.array([min_node_degree(other_edge, degree_dict) for other_edge in all_edges])

    if prob_type == "distance":
        if dist_mat is None:
            print("No distance matrix provided!")
        u_dist = np.array([edge_dist(edge, other_edge, dist_mat) for other_edge in all_edges])
        # Deal with 0 distances
        u_dist[u_dist == np.inf] = 0
        # Proximity parameter: positive = prefer far, 0 = uniform, negative = prefer near
        u_dist = u_dist ** (float(alpha))

    elif prob_type == "degree":
        if degree_dict is None:
            print("No degree dict provided!")
        # Degree priority parameter: positive = prefer hubs, 0 = uniform, negative = deprioritize hubs
        u_dist = e_degs ** (float(alpha))
        print(f"Num non-zero probs -- begin: {len([p for p in u_dist if p > 0])}")
        # Deal with 0 distances
        u_dist[u_dist == np.inf] = 0
        print(f"Num non-zero probs -- no inf degree: {len([p for p in u_dist if p > 0])}")


    # Zero out edge degrees that are too high or low
    u_dist[e_degs < min_edeg] = 0
    u_dist[e_degs > max_edeg] = 0
    u_dist[e_mnds < min_mnd] = 0
    u_dist[e_mnds > max_mnd] = 0
    print(f"Num non-zero probs -- bounded edeg and mnd: {len([p for p in u_dist if p > 0])}")

    if whitelist_pairs is None:
        # Zero out edges that aren't of the whitelist relation type
        if rel_whitelist is not None:
            in_whitelist_rel_mask = [(other_edge[-1]['edge'] in rel_whitelist) for other_edge in all_edges]
            u_dist *= in_whitelist_rel_mask
            print(f"Num non-zero probs -- no non-whitelist type: {len([p for p in u_dist if p > 0])}")

    # If whitelist pairs is specified, that takes precedent. Zero out edges that aren't whitelist pairs
    else:
        in_whitelist_pairs_mask = [((other_edge[0], other_edge[1]) in whitelist_pairs) for other_edge in all_edges]
        u_dist *= in_whitelist_pairs_mask

    # Avoid hitting the input edge
    if edge in all_edges:  # might not be the case in the contradictification pipeline
        edge_idx = all_edges.index(edge)
        u_dist[edge_idx] = 0

    # Replace NaNs with 0s
    u_dist = np.nan_to_num(u_dist)
    print(f"Num non-zero probs -- replace nan with 0: {len([p for p in u_dist if p > 0])}")

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
    if prob_type == "distance":
        for edge in edge_set:
            u_dist += prob_dist(edge,
                                all_edges,
                                dist_mat,
                                degree_dict,
                                prob_type,
                                graph,
                                alpha)
    else:
        u_dist = prob_dist(None, all_edges, dist_mat, degree_dict, prob_type, graph, alpha)

    # Make sure probability of test edges is 0
    print(f"Num non-zero probs: {np.count_nonzero(u_dist)}")
    edge_set_strs = set([str(edge) for edge in edge_set])  # to handle the dictionary which isn't hashable
    u_dist *= [(str(edge) not in edge_set_strs) for edge in all_edges]
    print(f"AFTER removing the test triples, num non-zero probs: {np.count_nonzero(u_dist)}")

    # e_count = 0
    # for edge in edge_set:
    #     if edge in all_edges:  # NEW CHANGE
    #         e_count += 1
    #         if e_count % 500 == 0:
    #             print(f"e_count: {e_count}")
    #         edge_idx = all_edges.index(edge)
    #         u_dist[edge_idx] = 0

    p_dist = u_dist / sum(u_dist)

    return p_dist


def random_split_list(in_list, val_frac):
    random.shuffle(in_list)
    k = int(good_round(len(in_list) * val_frac))
    return in_list[:k], in_list[k:]


def remove_E(G):
    E_free_edges = [e for e in G.edges(data=True) if e[-1]['edge'] not in ["E", "DaG", "CbG"]]
    print(f"New network without 'E' has len: {len(E_free_edges)}")
    return nx.MultiDiGraph(E_free_edges)


def filter_in_etype(G, edge_types):
    filter_in_edges = []
    for edge in edge_types:
        filter_in_edges += [e for e in G.edges(data=True) if e[-1]['edge'] == edge]

    print(f"New network with {edge_types} filtered in has len: {len(filter_in_edges)}")
    return nx.MultiDiGraph(filter_in_edges)


def randomize_edges(G, rel_whitelist):
    edge_types = set([r for _, _, r in G.edges(data='edge')])
    edge_types_nonwhite = edge_types.difference(rel_whitelist)
    new_edges = []
    for e in G.edges(data=True):
        edge_type = e[-1]['edge']
        if edge_type in rel_whitelist:
            new_edges.append(e)
        else:
            alternate_edge_types = edge_types_nonwhite.difference(edge_type)
            sampled_alternate = np.random.choice(list(alternate_edge_types), 1)
            new_edge = (e[0], e[1], {'edge': sampled_alternate[0]})
            new_edges.append(new_edge)

    print(f"New network with replaced relations has len: {len(new_edges)}")
    return nx.MultiDiGraph(new_edges)


def make_all_one_type(G, rel_whitelist):
    new_edges = []
    #for e in G.edges(data=True):
    #    edge_type = e[-1]['edge']
        #if edge_type in rel_whitelist:
        #    new_edges.append(e)
        #else:
        #    new_edge = (e[0], e[1], {'edge': 'blah'})
        #    new_edges.append(new_edge)

    #return nx.MultiDiGraph(new_edges)
    edges = G.edges(data=True)
    dr_dz_whitelist_pairs = set([(e[0], e[1]) for e in edges if e[-1]['edge'] in rel_whitelist])

    blah_edges = [(u, v, {'edge': "blah"}) for u, v, _ in edges]
    return nx.MultiDiGraph(blah_edges), dr_dz_whitelist_pairs


def remove_hubs(G, hub_size=500):
    degree_dict = dict(G.degree())
    large_deg_nodes = [node for node in degree_dict.keys() if degree_dict[node] > float(hub_size)]
    G2 = G.copy()
    G2.remove_nodes_from(large_deg_nodes)
    print(f"New network without hubs of degree {hub_size} or greater has len: {G2.number_of_edges()}")
    return nx.MultiDiGraph(G2.edges(data=True))



