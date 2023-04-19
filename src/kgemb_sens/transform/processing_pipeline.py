# -*- coding: utf-8 -*-

"""Main processing pipeline for manipulating the KGs before embedding."""

import os

import itertools
import networkx as nx
import numpy as np

from collections import Counter

from kgemb_sens.transform.contradiction_utilities import fill_with_contradictions, negative_completion, remove_contradictions
from kgemb_sens.transform.graph_utilities import prob_dist, prob_dist_from_list, random_split_list, undirect_multidigraph, edge_degree, min_node_degree
from kgemb_sens.utilities import good_round


def graph_processing_pipeline(G, i, params, out_dir, edge_names=None, SEED=1, G_undir=None, antonyms=None,
                              dist_mat=None, degree_dict=None, in_val_test_subset=None, replace_edges=False,
                              test_min_edeg=0, test_max_edeg=float("inf"), test_min_mnd=0, test_max_mnd=float("inf"),
                              rel_whitelist=None, dr_dz_whitelist_pairs=None):

    if G_undir is None:
        G_undir = undirect_multidigraph(G)

    print(f"Starting iteration {i + 1}")
    c = 1
    found_one = False
    G_con = None
    sparsified_subset, new_contradictions, removed_contradictions = None, None, None

    edges = list(G.edges(data=True, keys=True))
    e_degs = np.array([edge_degree(G_undir, other_edge, degree_dict) for other_edge in edges])
    e_mnds = np.array([min_node_degree(other_edge, degree_dict) for other_edge in edges])
    passes_deg_filters = ((e_degs >= test_min_edeg) * (e_degs <= test_max_edeg) * (e_mnds >= test_min_mnd) * (e_mnds <= test_max_mnd))
    whitelist_type = np.array([(edge[-1]['edge'] in rel_whitelist) for edge in edges])
    print(f"Num passes deg filters: {np.sum(passes_deg_filters)}")
    print(f"Num whitelist type: {np.sum(whitelist_type)}")
    num_valid_test_edges = np.sum(passes_deg_filters * whitelist_type)
    print(f"Num valid test edges: {num_valid_test_edges}")

    while not found_one:

        np.random.seed((i + 1) * c * SEED)

        if in_val_test_subset is None:
            if params["val_test_frac"] < 1:
                # TODO: Change to number of edges of specific type
                if rel_whitelist is None:
                    val_test_set_size = good_round(params["val_test_frac"] * G.number_of_edges())
                else:
                    # If there is a collection of dr_dz pairs (indicating the single relation type condition)
                    if dr_dz_whitelist_pairs is not None:
                        val_test_set_size = good_round(params["val_test_frac"] * len(dr_dz_whitelist_pairs))
                    else:
                        whitelist_count = 0
                        G_rel_counter = Counter([r for _, _, r in G.edges(data='edge')])
                        for rel in rel_whitelist:
                            whitelist_count += G_rel_counter[rel]
                        # TODO: change the count so it's 80/20 split of eligible edges (MND and edge degree)
                        # TODO 2: check that this all works at sparsity = 1. And with the
                        val_test_set_size = good_round(params["val_test_frac"] * min(num_valid_test_edges, whitelist_count))
                        print(f"Val test set size: {val_test_set_size}")
                        print(f"Num whitelist edges: {whitelist_count}")
                        print(f"Total number of edges {G.number_of_edges()}")
            else:
                val_test_set_size = int(params["val_test_frac"])

            probabilities = list(prob_dist(None, edges, dist_mat=None, degree_dict=degree_dict, prob_type="degree",
                                           graph=G_undir, alpha=params["vt_alpha"],
                                           min_edeg=test_min_edeg, max_edeg=test_max_edeg,
                                           min_mnd=test_min_mnd, max_mnd=test_max_mnd,
                                           rel_whitelist=rel_whitelist, whitelist_pairs=dr_dz_whitelist_pairs))
            print(f"OVERALL: Num non-zero probs for selecting test: {len([p for p in probabilities if p > 0])}")
            val_test_subset_idx = list(np.random.choice(len(edges), val_test_set_size, replace=False, p=probabilities))

            val_test_subset = []
            for idx in val_test_subset_idx:
                val_test_subset.append(edges[idx])

        else:  # Especially for debugging
            val_test_subset = in_val_test_subset

        val_subset, test_subset = random_split_list(val_test_subset, params["val_frac"])

        if params["prob_type"] == "distance":
            if dist_mat is None:
                dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(G_undir)))
            probabilities = prob_dist_from_list(val_test_subset,
                                                edges,
                                                dist_mat=dist_mat,
                                                prob_type="distance",
                                                alpha=params["alpha"])

        elif params["prob_type"] == "degree":
            if degree_dict is None:
                if params["flatten_kg"] == "True":
                    G_flat = nx.Graph(G_undir)
                    degree_dict = dict(G_flat.degree())
                else:
                    degree_dict = dict(G.degree())

            probabilities = prob_dist_from_list(val_test_subset,
                                                edges,
                                                degree_dict=degree_dict,
                                                prob_type="degree",
                                                graph=G_undir,
                                                alpha=params["alpha"])

        train_subset = edges[:]

        # Calculate number of edges adjacent to the test nodes
        test_node_set = set(list(itertools.chain(*[(u, v) for u, v, _, _ in val_test_subset])))
        edges_adj_to_test = [edge for edge in edges if (len(set([edge[0], edge[1]]).intersection(test_node_set)) > 0)]
        print("FINAL PRE-SPARSE STATS:")
        print(f"len val_test_subset: {len(val_test_subset)}")
        print(f"len edges adj_to_test: {len(edges_adj_to_test)}")

        if params["MODE"] == "sparsification":
            # parsified_set_size = good_round(params["sparsified_frac"] * G.number_of_edges())
            sparsified_set_size = good_round(params["sparsified_frac"] * (len(edges) - len(edges_adj_to_test)))
            print(f"SIZE OF SPARSIFIED SUBSET: {sparsified_set_size}")

            if np.count_nonzero(probabilities) > sparsified_set_size:
                found_one = True
            else:
                c += 1
                print("Not enough edges to sample!! Trying again")
                continue

            ##non_val_test_edges = list(set(np.arange(G.number_of_edges())).difference(set(val_test_subset_idx)))
            sparsified_subset_idx = list(
                np.random.choice(G.number_of_edges(), sparsified_set_size, replace=False, p=probabilities))
            sparsified_subset = []
            for idx in sparsified_subset_idx:
                sparsified_subset.append(edges[idx])

        elif params["MODE"] == "contrasparsify":
            found_one = True
            G_con, sampled_rel_edges, contradictory_edges = fill_with_contradictions(G, edge_names, val_test_subset,
                                                                                     params, G_undir=G_undir,
                                                                                     dist_mat=dist_mat,
                                                                                     degree_dict=degree_dict,
                                                                                     antonyms=antonyms,
                                                                                     replace_edges=replace_edges,
                                                                                     SEED=SEED)

            new_contradictions = sampled_rel_edges + contradictory_edges

            G_con, removed_contradictions = remove_contradictions(G_con, sampled_rel_edges, contradictory_edges,
                                                                          params["contra_remove_frac"], SEED=SEED)

            train_subset = list(G_con.edges(data=True, keys=True))

        # Need to check that the nodes and relations are found in the training too
        for val_test_edge in val_test_subset:
            train_subset.remove(val_test_edge)

        if params["MODE"] == "sparsification":
            #for sparsified_edge in sparsified_subset:
            #    # TODO: optimize this
            #    train_subset.remove(sparsified_edge)

            sparsified_subset_strs = set([str(edge) for edge in sparsified_subset])
            print(f"Len train subset before sparsification: {len(train_subset)}")
            train_subset = [train_edge for train_edge in train_subset if (str(train_edge) not in sparsified_subset_strs)]
            print(f"Len train subset after sparsification: {len(train_subset)}")

        nodes_in_train = set([item[0] for item in train_subset]).union(set([item[1] for item in train_subset]))
        relations_in_train = set([item[3]['edge'] for item in train_subset])

        test_not_in_train = False
        print(f"Test subset!! sample: {test_subset[:10]}")

        for u, v, k, r in test_subset:
            if (u not in nodes_in_train) or (v not in nodes_in_train):
                test_not_in_train = True
                break
            if r['edge'] not in relations_in_train:
                test_not_in_train = True
                break
        if test_not_in_train:
            c += 1
            print("Test edges or relations never seen in training, can't predict about them.")
            found_one = False
            continue

        # This is a hacky way to deal with PyKEEN bug. If only a single edge, need to duplicate so that the numpy array isn't
        # cast to 1D array when unpacking data. MultiDiGraph will allow duplicate edges to be created now, which will be removed
        # when data is loaded.
        if len(val_subset) == 1:
            (v_u, v_v, v_k, v_rd) = val_subset[0]
            val_subset.append((v_u, v_v, v_k + 1, v_rd))
        if len(test_subset) == 1:
            (t_u, t_v, t_k, t_rd) = test_subset[0]
            test_subset.append((t_u, t_v, t_k + 1, t_rd))

    if G_con is not None:
        G_out = G_con.copy()
    else:
        G_out = G.copy()

    if params["MODE"] == "sparsification":
        G_out.remove_edges_from(sparsified_subset)

    G_sparse = G_out.copy()

    G_out_test = nx.MultiDiGraph()
    G_out_test.add_edges_from(test_subset)
    print(f"Test subset 2!! sample: {list(G_out_test.edges(data=True))[:10]}")
    ##G_out_val = nx.MultiDiGraph()
    ##G_out_val.add_edges_from(val_subset)
    G_out.remove_edges_from(val_test_subset)
    G_out_train = G_out

    #train_conditions_id = f"{params['MODE']}_alpha{params['alpha']}_probtype{params['prob_type']}_flat{params['flatten_kg']}_sparsefrac{params['sparsified_frac']}_negCompFrac{params['neg_completion_frac']}_contraFrac{params['contradiction_frac']}_contraRemFrac{params['contra_remove_frac']}_vtfrac{params['val_test_frac']}_vtalpha{params['vt_alpha']}"
    train_conditions_id = f"sparsefrac{params['sparsified_frac']}_contraFrac{params['contradiction_frac']}_contraRemFrac{params['contra_remove_frac']}_vtalpha{params['vt_alpha']}"
    ##train_conditions_id += f"_remEfilter{params['remove_E_filter']}_filtInAnto{params['filter_in_antonyms']}_randRelations{params['randomize_relations']}_singRelation{params['single_relation']}_hubRemThresh{params['hub_remove_thresh']}_testMinEdeg{params['test_min_edeg']}_testMaxEdeg{params['test_max_edeg']}_replaceEdges{params['replace_edges']}"
    os.makedirs(out_dir, exist_ok=True)

    new_test_path = f"{out_dir}/test_{train_conditions_id}.tsv"
    print(f"Saving test triples to: {new_test_path}")
    G_out_test_df = nx.to_pandas_edgelist(G_out_test)[['source', 'edge', 'target']]
    G_out_test_df.to_csv(new_test_path, sep='\t', header=False, index=False)

    ##new_val_path = f"{out_dir}/val_{train_conditions_id}.tsv"
    ##G_out_val_df = nx.to_pandas_edgelist(G_out_val)[['source', 'edge', 'target']]
    ##G_out_val_df.to_csv(new_val_path, sep='\t', header=False, index=False)

    new_train_path = f"{out_dir}/train_{train_conditions_id}.tsv"
    print(f"Saving train triples to: {new_train_path}")
    G_out_train_df = nx.to_pandas_edgelist(G_out_train)[['source', 'edge', 'target']]
    G_out_train_df.to_csv(new_train_path, sep='\t', header=False, index=False)

    ##return [new_train_path, new_val_path, new_test_path]

    if params["MODE"] == "sparsification":
        G_out = G_sparse
    else:
        G_out = G_con

    return (new_train_path, new_test_path), train_conditions_id, (train_subset, test_subset, sparsified_subset, new_contradictions, removed_contradictions), G_out


def simplified_graph_processing_pipeline(G, i, params, out_dir, SEED=1, G_undir=None, dist_mat=None, degree_dict=None,
                                         in_val_test_subset=None, test_min_edeg=0, test_max_edeg=float("inf"),
                                         test_min_mnd=0, test_max_mnd=float("inf"), rel_whitelist=None,
                                         dr_dz_whitelist_pairs=None):

    if G_undir is None:
        G_undir = undirect_multidigraph(G)

    print(f"Starting iteration {i + 1}")
    c = 1
    found_one = False

    edges = list(G.edges(data=True, keys=True))
    e_degs = np.array([edge_degree(G_undir, other_edge, degree_dict) for other_edge in edges])
    e_mnds = np.array([min_node_degree(other_edge, degree_dict) for other_edge in edges])
    passes_deg_filters = ((e_degs >= test_min_edeg) * (e_degs <= test_max_edeg) * (e_mnds >= test_min_mnd) * (e_mnds <= test_max_mnd))
    whitelist_type = np.array([(edge[-1]['edge'] in rel_whitelist) for edge in edges])
    print(f"Num passes deg filters: {np.sum(passes_deg_filters)}")
    print(f"Num whitelist type: {np.sum(whitelist_type)}")
    num_valid_test_edges = np.sum(passes_deg_filters * whitelist_type)
    print(f"Num valid test edges: {num_valid_test_edges}")

    while not found_one:

        found_one = True
        np.random.seed((i + 1) * c * SEED)

        if in_val_test_subset is None:
            if params["val_test_frac"] < 1:
                # TODO: Change to number of edges of specific type
                if rel_whitelist is None:
                    val_test_set_size = good_round(params["val_test_frac"] * G.number_of_edges())
                else:
                    # If there is a collection of dr_dz pairs (indicating the single relation type condition)
                    if dr_dz_whitelist_pairs is not None:
                        val_test_set_size = good_round(params["val_test_frac"] * len(dr_dz_whitelist_pairs))
                    else:
                        whitelist_count = 0
                        G_rel_counter = Counter([r for _, _, r in G.edges(data='edge')])
                        for rel in rel_whitelist:
                            whitelist_count += G_rel_counter[rel]
                        # TODO: change the count so it's 80/20 split of eligible edges (MND and edge degree)
                        # TODO 2: check that this all works at sparsity = 1. And with the
                        val_test_set_size = good_round(params["val_test_frac"] * min(num_valid_test_edges, whitelist_count))
                        print(f"Val test set size: {val_test_set_size}")
                        print(f"Num whitelist edges: {whitelist_count}")
                        print(f"Total number of edges {G.number_of_edges()}")
            else:
                val_test_set_size = int(params["val_test_frac"])

            probabilities = list(prob_dist(None, edges, dist_mat=None, degree_dict=degree_dict, prob_type="degree",
                                           graph=G_undir, alpha=params["vt_alpha"],
                                           min_edeg=test_min_edeg, max_edeg=test_max_edeg,
                                           min_mnd=test_min_mnd, max_mnd=test_max_mnd,
                                           rel_whitelist=rel_whitelist, whitelist_pairs=dr_dz_whitelist_pairs))
            print(f"OVERALL: Num non-zero probs for selecting test: {len([p for p in probabilities if p > 0])}")
            val_test_subset_idx = list(np.random.choice(len(edges), val_test_set_size, replace=False, p=probabilities))

            val_test_subset = []
            for idx in val_test_subset_idx:
                val_test_subset.append(edges[idx])

        else:  # Especially for debugging
            val_test_subset = in_val_test_subset

        val_subset, test_subset = random_split_list(val_test_subset, params["val_frac"])

        if params["prob_type"] == "distance":
            if dist_mat is None:
                dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(G_undir)))
            probabilities = prob_dist_from_list(val_test_subset,
                                                edges,
                                                dist_mat=dist_mat,
                                                prob_type="distance",
                                                alpha=params["alpha"])

        elif params["prob_type"] == "degree":
            if degree_dict is None:
                if params["flatten_kg"] == "True":
                    G_flat = nx.Graph(G_undir)
                    degree_dict = dict(G_flat.degree())
                else:
                    degree_dict = dict(G.degree())

            probabilities = prob_dist_from_list(val_test_subset,
                                                edges,
                                                degree_dict=degree_dict,
                                                prob_type="degree",
                                                graph=G_undir,
                                                alpha=params["alpha"])

        train_subset = edges[:]

        # Calculate number of edges adjacent to the test nodes
        test_node_set = set(list(itertools.chain(*[(u, v) for u, v, _, _ in val_test_subset])))
        edges_adj_to_test = [edge for edge in edges if (len(set([edge[0], edge[1]]).intersection(test_node_set)) > 0)]
        print("FINAL PRE-SPARSE STATS:")
        print(f"len val_test_subset: {len(val_test_subset)}")
        print(f"len edges adj_to_test: {len(edges_adj_to_test)}")

        # Need to check that the nodes and relations are found in the training too
        for val_test_edge in val_test_subset:
            train_subset.remove(val_test_edge)

        nodes_in_train = set([item[0] for item in train_subset]).union(set([item[1] for item in train_subset]))
        relations_in_train = set([item[3]['edge'] for item in train_subset])

        test_not_in_train = False
        print(f"Test subset!! sample: {test_subset[:10]}")

        for u, v, k, r in test_subset:
            if (u not in nodes_in_train) or (v not in nodes_in_train):
                test_not_in_train = True
                break
            if r['edge'] not in relations_in_train:
                test_not_in_train = True
                break
        if test_not_in_train:
            c += 1
            print("Test edges or relations never seen in training, can't predict about them.")
            found_one = False
            continue

        # This is a hacky way to deal with PyKEEN bug. If only a single edge, need to duplicate so that the numpy array isn't
        # cast to 1D array when unpacking data. MultiDiGraph will allow duplicate edges to be created now, which will be removed
        # when data is loaded.
        if len(val_subset) == 1:
            (v_u, v_v, v_k, v_rd) = val_subset[0]
            val_subset.append((v_u, v_v, v_k + 1, v_rd))
        if len(test_subset) == 1:
            (t_u, t_v, t_k, t_rd) = test_subset[0]
            test_subset.append((t_u, t_v, t_k + 1, t_rd))


    G_out = G.copy()
    G_out_test = nx.MultiDiGraph()
    G_out_test.add_edges_from(test_subset)
    print(f"Test subset 2!! sample: {list(G_out_test.edges(data=True))[:10]}")
    ##G_out_val = nx.MultiDiGraph()
    ##G_out_val.add_edges_from(val_subset)
    G_out.remove_edges_from(val_test_subset)
    G_out_train = G_out

    #train_conditions_id = f"{params['MODE']}_alpha{params['alpha']}_probtype{params['prob_type']}_flat{params['flatten_kg']}_sparsefrac{params['sparsified_frac']}_negCompFrac{params['neg_completion_frac']}_contraFrac{params['contradiction_frac']}_contraRemFrac{params['contra_remove_frac']}_vtfrac{params['val_test_frac']}_vtalpha{params['vt_alpha']}"
    train_conditions_id = f"sparsefrac{params['sparsified_frac']}_contraFrac{params['contradiction_frac']}_contraRemFrac{params['contra_remove_frac']}_vtalpha{params['vt_alpha']}"
    ##train_conditions_id += f"_remEfilter{params['remove_E_filter']}_filtInAnto{params['filter_in_antonyms']}_randRelations{params['randomize_relations']}_singRelation{params['single_relation']}_hubRemThresh{params['hub_remove_thresh']}_testMinEdeg{params['test_min_edeg']}_testMaxEdeg{params['test_max_edeg']}_replaceEdges{params['replace_edges']}"
    os.makedirs(out_dir, exist_ok=True)

    new_test_path = f"{out_dir}/test_{train_conditions_id}.tsv"
    print(f"Saving test triples to: {new_test_path}")
    G_out_test_df = nx.to_pandas_edgelist(G_out_test)[['source', 'edge', 'target']]
    G_out_test_df.to_csv(new_test_path, sep='\t', header=False, index=False)

    ##new_val_path = f"{out_dir}/val_{train_conditions_id}.tsv"
    ##G_out_val_df = nx.to_pandas_edgelist(G_out_val)[['source', 'edge', 'target']]
    ##G_out_val_df.to_csv(new_val_path, sep='\t', header=False, index=False)

    new_train_path = f"{out_dir}/train_{train_conditions_id}.tsv"
    print(f"Saving train triples to: {new_train_path}")
    G_out_train_df = nx.to_pandas_edgelist(G_out_train)[['source', 'edge', 'target']]
    G_out_train_df.to_csv(new_train_path, sep='\t', header=False, index=False)

    return (new_train_path, new_test_path), train_conditions_id, (train_subset, test_subset), G_out
