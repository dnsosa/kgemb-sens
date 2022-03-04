# -*- coding: utf-8 -*-

"""Utilities for contradiction-based perturbations of KG."""

import itertools
import numpy as np

from kgemb_sens.transform.graph_utilities import prob_dist_from_list


def find_all_valid_negations(G):
    all_valid_negations = []
    all_rels = set([r for _, _, r in G.edges(data='edge')])  # NOTE DOING FOR ALL RELS

    for pair in itertools.permutations(G.nodes(), 2):
        for edge_name in all_rels:
            candidate_edge = (pair[0], pair[1], edge_name)
            if candidate_edge not in G.edges(data='edge'):
                negated_edge = (pair[0], pair[1], f"NOT-{edge_name}")
                all_valid_negations.append(negated_edge)

    return all_valid_negations


def negative_completion(G, all_valid_negations, neg_completion_frac):
    G_complete = G.copy()

    n_negations = len(all_valid_negations)
    sampled_negations_idx = np.random.choice(n_negations, min(round(neg_completion_frac * G.number_of_edges()), n_negations), replace=False)
    sampled_negations = []
    for idx in sampled_negations_idx:
        sampled_negations.append(all_valid_negations[idx])

    G_complete.add_edges_from([(u, v, {'edge': r}) for u, v, r in sampled_negations])  # Let it fill in the key?

    return G_complete


def generate_converse_edges_from(edge_list):
    def flip_relation(relation):
        if relation.startswith("NOT-"):
            converse_relation = relation.split("NOT-")[1]
        else:
            converse_relation = f"NOT-{relation}"
        return converse_relation

    out_list = []
    for e in edge_list:
        if len(e) > 2:
            if type(e[-1]) == dict:
                if 'edge' in e[-1] and len(e[-1].keys()) == 1:
                    # TODO: Revisit if we ever want to deal with multiple edge attributes at once. Need to just update relevant k,v pair. This will be wrong in that case.
                    new_relation = {'edge': flip_relation(e[-1]['edge'])}
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
                new_relation = flip_relation(e[-1])
                new_edge = (e[0], e[1], new_relation)  # HOPE TO NEVER GET THIS FORMAT
        elif len(e) == 2:
            new_edge = e
        else:
            assert "Not an edge! (len < 2)"
            return None

        out_list.append(new_edge)

    return out_list


def fill_with_contradictions(G, edge_names, val_test_subset, params, all_pairs_lens=None, degree_dict=None):
    G_contra = G.copy()

    all_sampled_rel_edges = []
    all_contradictory_edges = []

    for edge_name in edge_names:
        rel_edges = [e for e in G_contra.edges(data=True, keys=True) if e[3]['edge'] == edge_name]
        n_rel_edges = len(rel_edges)

        if params["prob_type"] == "distance":
            probabilities = prob_dist_from_list(val_test_subset,
                                                rel_edges,
                                                dist_mat=all_pairs_lens,
                                                prob_type="distance",
                                                alpha=params["alpha"])
        elif params["prob_type"] == "degree":
            probabilities = prob_dist_from_list(val_test_subset,
                                                rel_edges,
                                                degree_dict=degree_dict,
                                                prob_type="degree",
                                                graph=G_contra,
                                                alpha=params["alpha"])
        nz_probs = np.count_nonzero(probabilities)
        # print(f"For edge_name: {edge_name}, # of non-zeros probs found: {np.count_nonzero(probabilities)}. Requesting: {round(params['contradiction_frac'] * n_rel_edges)}")
        sampled_rel_edges_idx = np.random.choice(n_rel_edges,
                                                 min(round(params["contradiction_frac"] * n_rel_edges), nz_probs),
                                                 replace=False, p=probabilities)
        sampled_rel_edges = []
        for idx in sampled_rel_edges_idx:
            sampled_rel_edges.append(rel_edges[idx])

        contradictory_edges = generate_converse_edges_from(sampled_rel_edges)
        contra_keys = G_contra.add_edges_from(contradictory_edges)  # TEST!
        contradictory_edges = [(u, v, contra_keys[i], r) for i, (u, v, _, r) in enumerate(contradictory_edges)]

        all_sampled_rel_edges += sampled_rel_edges
        all_contradictory_edges += contradictory_edges

    return G_contra, all_sampled_rel_edges, all_contradictory_edges  ###, contra_keys


def remove_contradictions(G, edge_set_1, edge_set_2, edges_not_to_remove, contra_remove_frac):
    ### edges_not_to_remove_reformat = [(u,v,r['edge']) for u,v,r in edges_not_to_remove]

    G_contra_remove = G.copy()
    if len(edge_set_1) != len(edge_set_2):
        print("Different sized edge sets!! Something went wrong?")
        return None

    n_contras = len(edge_set_1)
    sampled_contra_idx = np.random.choice(n_contras, round(contra_remove_frac * n_contras), replace=False)
    sampled_contras = []
    for idx in sampled_contra_idx:
        if edge_set_1[idx] not in edges_not_to_remove:
            sampled_contras.append(edge_set_1[idx])
        if edge_set_2[idx] not in edges_not_to_remove:
            sampled_contras.append(edge_set_2[idx])

    G_contra_remove.remove_edges_from(sampled_contras)

    return G_contra_remove, sampled_contras