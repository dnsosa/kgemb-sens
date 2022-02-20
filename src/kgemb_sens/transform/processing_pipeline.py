# -*- coding: utf-8 -*-

"""Main processing pipeline for manipulating the KGs before embedding."""

import networkx as nx
import numpy as np

from kgemb_sens.transform.contradiction_utilities import fill_with_contradictions, negative_completion, remove_contradictions
from kgemb_sens.transform.graph_utilities import prob_dist_from_list, random_split_list


def graph_processing_pipeline(G, i, params, out_dir, all_valid_negations=None, edge_names=None, SEED=1):
    print(f"Starting iteration {i + 1}")
    c = 1
    found_one = False
    G_con = None
    G_con_rem = None
    sparsified_subset, new_contradictions, removed_contradictions = None, None, None

    if params["neg_completion_frac"] > 0:
        #print("negative completion path")
        #print(f"Before neg comp: {G.number_of_edges()}")
        G = negative_completion(G, all_valid_negations, params["neg_completion_frac"])
        #print(f"AFTER neg comp: {G.number_of_edges()}")

    edges = list(G.edges(data=True, keys=True))

    while not found_one:

        np.random.seed((i + 1) * c * SEED)

        if params["val_test_frac"] < 1:
            val_test_set_size = round(params["val_test_frac"] * G.number_of_edges())
        else:
            val_test_set_size = int(params["val_test_frac"])

        val_test_subset_idx = list(np.random.choice(G.number_of_edges(), val_test_set_size, replace=False))
        val_test_subset = []

        for idx in val_test_subset_idx:
            val_test_subset.append(edges[idx])
        val_subset, test_subset = random_split_list(val_test_subset, params["val_frac"])

        if params["prob_type"] == "distance":
            all_pairs_lens = dict(nx.all_pairs_bellman_ford_path_length(G.to_undirected()))
            probabilities = prob_dist_from_list(val_test_subset,
                                                edges,
                                                dist_mat=all_pairs_lens,
                                                prob_type="distance",
                                                alpha=params["alpha"])
            degree_dict = None

        elif params["prob_type"] == "degree":
            if params["flatten_kg"] == "True":
                G_flat = nx.Graph(G.to_undirected())
                degree_dict = dict(G_flat.degree)
            else:
                degree_dict = dict(G.degree)

            probabilities = prob_dist_from_list(val_test_subset,
                                                edges,
                                                degree_dict=degree_dict,
                                                prob_type="degree",
                                                graph=G,
                                                alpha=params["alpha"])
            all_pairs_lens = None

        train_subset = edges[:]

        if params["MODE"] == "sparsification":
            sparsified_set_size = round(params["sparsified_frac"] * G.number_of_edges())

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

        elif params["MODE"] in ["contradictification", "contrasparsify"]:
            found_one = True  # Think about this??? What if contrasparsify goes too far?

            G_con, sampled_rel_edges, contradictory_edges = fill_with_contradictions(G, edge_names, val_test_subset,
                                                                                     params,
                                                                                     all_pairs_lens=all_pairs_lens,
                                                                                     degree_dict=degree_dict)
            new_contradictions = sampled_rel_edges + contradictory_edges
            train_subset = list(G_con.edges(data=True, keys=True))

            if params["MODE"] == "contrasparsify":
                G_con_rem, removed_contradictions = remove_contradictions(G_con, sampled_rel_edges, contradictory_edges,
                                                                          val_test_subset,
                                                                          params["contra_remove_frac"])
                train_subset = list(G_con_rem.edges(data=True, keys=True))

        # Need to check that the nodes and relations are found in the training too
        for val_test_edge in val_test_subset:
            train_subset.remove(val_test_edge)

        if params["MODE"] == "sparsification":
            for sparsified_edge in sparsified_subset:
                train_subset.remove(sparsified_edge)

        nodes_in_train = set([item[0] for item in train_subset]).union(set([item[1] for item in train_subset]))
        relations_in_train = set([item[3]['edge'] for item in train_subset])

        test_not_in_train = False
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
    elif G_con_rem is not None:
        G_out = G_con_rem.copy()
    else:
        G_out = G.copy()

    if params["MODE"] == "sparsification":
        G_out.remove_edges_from(sparsified_subset)

    G_out_test = nx.MultiDiGraph()
    G_out_test.add_edges_from(test_subset)
    ##G_out_val = nx.MultiDiGraph()
    ##G_out_val.add_edges_from(val_subset)
    G_out.remove_edges_from(val_test_subset)
    G_out_train = G_out

    train_conditions_id = f"{params['MODE']}_alpha{params['alpha']}_probtype{params['prob_type']}_flat{params['flatten_kg']}_sparsefrac{params['sparsified_frac']}_negCompFrac{params['neg_completion_frac']}_contraFrac{params['contradiction_frac']}_contraRemFrac{params['contra_remove_frac']}_vtfrac{params['val_test_frac']}"

    new_test_path = f"{out_dir}/test_{train_conditions_id}.tsv"
    G_out_test_df = nx.to_pandas_edgelist(G_out_test)[['source', 'edge', 'target']]
    G_out_test_df.to_csv(new_test_path, sep='\t', header=False, index=False)

    ##new_val_path = f"{out_dir}/val_{train_conditions_id}.tsv"
    ##G_out_val_df = nx.to_pandas_edgelist(G_out_val)[['source', 'edge', 'target']]
    ##G_out_val_df.to_csv(new_val_path, sep='\t', header=False, index=False)

    new_train_path = f"{out_dir}/train_{train_conditions_id}.tsv"
    G_out_train_df = nx.to_pandas_edgelist(G_out_train)[['source', 'edge', 'target']]
    G_out_train_df.to_csv(new_train_path, sep='\t', header=False, index=False)

    ##return [new_train_path, new_val_path, new_test_path]

    return (new_train_path, new_test_path), train_conditions_id, (train_subset, test_subset, sparsified_subset, new_contradictions, removed_contradictions), G_con

