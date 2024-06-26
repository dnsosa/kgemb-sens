# -*- coding: utf-8 -*-

"""Utilities for contradiction-based perturbations of KG."""

import itertools
import numpy as np

from kgemb_sens.transform.graph_utilities import prob_dist_from_list, undirect_multidigraph
from kgemb_sens.utilities import good_round


def remove_prefix_nots(word):
    return word.split("NOT-")[-1]


def find_all_valid_negations(G):
    # TODO: This will create a problem if there are already 'NOT-...' edges in the starting graph. Can ignore for now.
    all_valid_negations = []
    all_rels = set([r for _, _, r in G.edges(data='edge')])  # NOTE DOING FOR ALL RELS

    for pair in itertools.permutations(G.nodes(), 2):
        for edge_name in all_rels:
            candidate_edge = (pair[0], pair[1], edge_name)
            if candidate_edge not in G.edges(data='edge'):
                negated_edge = (pair[0], pair[1], f"NOT-{edge_name}")
                all_valid_negations.append(negated_edge)

    return all_valid_negations


def negative_completion(G, all_valid_negations, edge_names, neg_completion_frac, SEED=None):
    np.random.seed(SEED)

    G_complete = G.copy()

    if edge_names is None:
        edge_names = list(set([remove_prefix_nots(r) for _, _, _, r in G_complete.edges(data='edge', keys=True)]))

    for edge_name in edge_names:
        rel_edges = [e for e in G.edges(data=True, keys=True) if e[-1]['edge'] == edge_name]
        n_rel_edges = len(rel_edges)
        rel_neg_edges = [e for e in all_valid_negations if e[-1] == f"NOT-{edge_name}"]  # NOTE: expected format will be like this
        n_rel_neg_edges = len(rel_neg_edges)

        sampled_negations_idx = np.random.choice(n_rel_neg_edges, min(good_round(neg_completion_frac * n_rel_edges), n_rel_neg_edges), replace=False)
        sampled_negations = []
        for idx in sampled_negations_idx:
            sampled_negations.append(rel_neg_edges[idx])

        G_complete.add_edges_from([(u, v, {'edge': r}) for u, v, r in sampled_negations])  # Let it fill in the key

    return G_complete


def generate_converse_edges_from(edge_list, antonym_dict=None):
    def flip_relation(relation, antonym_lookup):
        if antonym_dict is None:
            if relation.startswith("NOT-"):
                converse_relation = relation.split("NOT-")[1]
            else:
                converse_relation = f"NOT-{relation}"
        else:
            if relation in antonym_lookup:
                converse_relation = antonym_lookup[relation]
            else:
                converse_relation = relation

        return converse_relation

    out_list = []
    for e in edge_list:
        if len(e) > 2:
            if type(e[-1]) == dict:
                if 'edge' in e[-1] and len(e[-1].keys()) == 1:
                    # TODO: Revisit if we ever want to deal with multiple edge attributes at once. Need to just update relevant k,v pair. This will be wrong in that case.
                    new_relation = {'edge': flip_relation(e[-1]['edge'], antonym_dict)}
                else:
                    new_relation = e[-1]

                if len(e) == 3:
                    new_edge = (e[0], e[1], new_relation)
                elif len(e) == 4:
                    new_edge = (e[0], e[1], None, new_relation)
                else:
                    assert "Weird format! (len > 4)"
                    return None
            else:
                print("WARNING: You're flipping an edge where the attribute isn't a dictionary... may cause confusion.")
                new_relation = flip_relation(e[-1], antonym_dict)
                new_edge = (e[0], e[1], new_relation)  # HOPE TO NEVER GET THIS FORMAT
        elif len(e) == 2:
            new_edge = e
        else:
            assert "Not an edge! (len < 2)"
            return None

        out_list.append(new_edge)

    return out_list


def fill_with_contradictions(G, edge_names, val_test_subset, params, G_undir=None, dist_mat=None, degree_dict=None, antonyms=None, replace_edges=False, SEED=None):
    np.random.seed(SEED)
    antonym_dict = None
    if antonyms is None:
        edge_names = set([remove_prefix_nots(edge_name) for edge_name in edge_names])
    else:
        antonym_dict = {}
        for pair in antonyms:
            antonym_dict[pair[0]] = pair[1]
            antonym_dict[pair[1]] = pair[0]
        edge_names = set(list(antonym_dict.keys()))

    if G_undir is None:
        G_undir = undirect_multidigraph(G)

    G_contra = G.copy()

    all_sampled_rel_edges = []
    all_contradictory_edges = []

    for edge_name in edge_names:
        if antonyms is None:
            rel_edges = [e for e in G_contra.edges(data=True, keys=True) if ((e[-1]['edge'] == edge_name) or (e[-1]['edge'] == f"NOT-{edge_name}"))]
        else:
            rel_edges = [e for e in G_contra.edges(data=True, keys=True) if (e[-1]['edge'] == edge_name)]

        n_rel_edges = len(rel_edges)

        if n_rel_edges == 0:
            continue

        print(f"Filling with contras... now adding {edge_name} contras....")

        if params["prob_type"] == "distance":
            probabilities = prob_dist_from_list(val_test_subset,
                                                rel_edges,
                                                dist_mat=dist_mat,
                                                prob_type="distance",
                                                graph=G_undir,
                                                alpha=params["alpha"])
        elif params["prob_type"] == "degree":
            print("calculating probs for fill with contra....")
            probabilities = prob_dist_from_list(val_test_subset,
                                                rel_edges,
                                                degree_dict=degree_dict,
                                                prob_type="degree",
                                                graph=G_undir,
                                                alpha=params["alpha"])
        nz_probs = np.count_nonzero(probabilities)

        # This prevents the test edge from being sampled and contradicted
        sampled_rel_edges_idx = np.random.choice(n_rel_edges,
                                                 min(good_round(params["contradiction_frac"] * n_rel_edges), nz_probs),
                                                 replace=False, p=probabilities)
        sampled_rel_edges = []
        for idx in sampled_rel_edges_idx:
            sampled_rel_edges.append(rel_edges[idx])

        all_contradictory_edges += generate_converse_edges_from(sampled_rel_edges, antonym_dict=antonym_dict)
        all_sampled_rel_edges += sampled_rel_edges

    # Need to update the keys of the newly added edges
    contra_keys = G_contra.add_edges_from(all_contradictory_edges)
    if replace_edges:
        G_contra.remove_edges_from(all_sampled_rel_edges)
    all_contradictory_edges = [(u, v, contra_keys[i], r) for i, (u, v, _, r) in enumerate(all_contradictory_edges)]

    return G_contra, all_sampled_rel_edges, all_contradictory_edges


def remove_contradictions(G, edge_set_1, edge_set_2, contra_remove_frac, SEED=None):
    np.random.seed(SEED)

    # Note: no risk of removing a test edge, because it will never be contradicted
    G_contra_remove = G.copy()
    if len(edge_set_1) != len(edge_set_2):
        print("Different sized edge sets!! Something went wrong?")
        return None

    sampled_contras = []
    edge_names = set([remove_prefix_nots(r['edge']) for _, _, _, r in edge_set_1])
    for edge_name in edge_names:
        rel_edges_idxs = [i for i, e in enumerate(edge_set_1) if (remove_prefix_nots(e[-1]['edge']) == edge_name)]
        n_rel_edges = len(rel_edges_idxs)

        if n_rel_edges == 0:
            continue

        sampled_contra_idx = np.random.choice(rel_edges_idxs, good_round(contra_remove_frac * n_rel_edges), replace=False)

        for idx in sampled_contra_idx:
            sampled_contras.append(edge_set_1[idx])
            sampled_contras.append(edge_set_2[idx])

    G_contra_remove.remove_edges_from(sampled_contras)

    return G_contra_remove, sampled_contras