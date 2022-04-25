# -*- coding: utf-8 -*-

"""Run the embedding pipeline."""

import networkx as nx
import numpy as np
import pandas as pd

from pykeen.pipeline import pipeline
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df

from kgemb_sens.analyze.metrics import calc_edge_input_statistics, calc_network_input_statistics, calc_output_statistics
from kgemb_sens.transform.graph_utilities import undirect_multidigraph


def run_embed_pipeline(data_paths, i, params, train_conditions_id, G, test_subset,
                       degree_dict=None, G_undir=None, rel_whitelist=None):

    # Note rel_whitelist is pretty much redundant, since it's taken care of in creating the test set

    if G_undir is None:
        G_undir = undirect_multidigraph(G)

    if len(data_paths) == 2:
        new_train_path, new_test_path = data_paths
    elif len(data_paths) == 3:
        new_train_path, new_val_path, new_test_path = data_paths

    train_df = pd.read_csv(new_train_path, sep="\t", header=None)
    test_df = pd.read_csv(new_test_path, sep="\t", header=None)
    train_df.columns = ["source", "edge", "target"]
    test_df.columns = ["source", "edge", "target"]
    G_train = nx.from_pandas_edgelist(train_df, source="source", target="target", edge_attr="edge", create_using=nx.MultiDiGraph)
    G_test = nx.from_pandas_edgelist(test_df, source="source", target="target", edge_attr="edge", create_using=nx.MultiDiGraph)

    print(f"Num nodes in test: {G_test.number_of_nodes()}")
    print(f"Checking that the test nodes are in the train triples: {len(set(G_test.nodes()).intersection(set(G_train.nodes()))) == G_test.number_of_nodes()}")

    print("Now running pipeline...")
    result = pipeline(
        training=new_train_path,
        ##validation=new_val_path,
        testing=new_test_path,
        ##evaluation_relation_whitelist=rel_whitelist,
        model=params["model_name"],
        # Training configuration
        training_kwargs=dict(
            num_epochs=params["n_epochs"],
            use_tqdm_batch=False,
        ),
        negative_sampler_kwargs=dict(
            num_negs_per_pos=params["n_negatives"]
        ),
        # Runtime configuration
        random_seed=1235
    )
    print("Pipeline finished.")

    run_id = f"{train_conditions_id}_model{params['model_name']}"

    test_triple = pd.read_csv(new_test_path, header=None, sep="\t").drop_duplicates()
    test_triple.columns = ["source", "edge", "target"]
    # u, r, v = test_triple['source'][0], test_triple['edge'][0], test_triple['target'][0]

    print("Calculating test edge statistics...")
    edge_min_node_degrees, edge_rel_counts, e_degs = [], [], []
    for test_edge in test_subset:

        print(f"Test edge: {test_edge}")
        edge_min_node_degree, edge_rel_count, e_deg = calc_edge_input_statistics(G, test_edge, degree_dict, G_undir=G_undir)
        edge_min_node_degrees.append(edge_min_node_degree)
        edge_rel_counts.append(edge_rel_count)
        e_degs.append(e_deg)

    print(f"edge_min_node_degrees: {edge_min_node_degrees}")
    avg_edge_min_node_degrees = np.average(edge_min_node_degrees)

    print(f"edge_rel_counts: {edge_rel_counts}")
    avg_edge_rel_counts = np.average(edge_rel_counts)

    print(f"e_degs: {e_degs}")
    avg_e_degs = np.average(e_degs)
    print("Test edge statistics done.")

    print("Calculating input network statistics..")
    net_stats = calc_network_input_statistics(G, calc_expensive=False, G_undir=G_undir)
    n_ent_network, n_rel_network, n_triples, n_conn_comps, med_rel_count, min_rel_count, rel_entropy, ent_entropy = net_stats
    print("Done with network statistics.")

    # head_prediction_df = get_head_prediction_df(result.model, str(r), str(v), triples_factory=result.training)
    # tail_prediction_df = get_tail_prediction_df(result.model, str(u), str(r), triples_factory=result.training)
    # head_deg_rank_corr = calc_output_statistics(list(head_prediction_df.head_label), degree_dict)
    # tail_deg_rank_corr = calc_output_statistics(list(tail_prediction_df.tail_label), degree_dict)

    results_dict = {'Dataset': params["dataset"],
                    'PCNet_filter': params["pcnet_filter"],
                    'Model_name': params["model_name"],
                    'Sparsified_frac': params["sparsified_frac"],
                    "VT_alpha": params["vt_alpha"],
                    'Alpha': params["alpha"],
                    'Prob_type': params["prob_type"],
                    'Flatten_kg': params["flatten_kg"],
                    'remove_E_filter': params["remove_E_filter"],
                    'filter_in_antonyms': params["filter_in_antonyms"],
                    'randomize_relations': params["randomize_relations"],
                    'single_relation': params["single_relation"],
                    'hub_remove_thresh': params["hub_remove_thresh"],
                    'test_min_edeg': params["test_min_edeg"],
                    'test_max_edeg': params["test_max_edeg"],
                    'test_min_mnd': params["test_min_mnd"],
                    'test_max_mnd': params["test_max_mnd"],
                    'replace_edges': params["replace_edges"],
                    'Neg_Completion_Frac': params["neg_completion_frac"],
                    'Contradiction_Frac': params["contradiction_frac"],
                    'Contra_Remove_Frac': params["contra_remove_frac"],
                    'MODE': params["MODE"],
                    'PSL': params["psl"],
                    'PSL contradictions': params["psl_contras"],
                    'full_product': params["full_product"],
                    'Num_resamples': params["n_resample"],
                    'Num_negative_samples': params["n_negatives"],
                    # 'Val_test_subset_idx': str(val_test_subset_idx),
                    'Num_epochs': params["n_epochs"],
                    'Repurposing evaluation': params["repurposing_evaluation"],
                    'Run': i,
                    'Run_ID': run_id,
                    'AMRI': result.metric_results.get_metric('adjusted_mean_rank_index'),
                    'Hits@1': result.metric_results.get_metric('hits@1'),
                    'Hits@3': result.metric_results.get_metric('hits@3'),
                    'Hits@5': result.metric_results.get_metric('hits@5'),
                    'Hits@10': result.metric_results.get_metric('hits@10'),
                    'MR': result.metric_results.get_metric('mean_rank'),
                    'MRR': result.metric_results.get_metric('mean_reciprocal_rank'),
                    #'Head Deg Rank Corr': head_deg_rank_corr,
                    #'Tail Deg Rank Corr': tail_deg_rank_corr,
                    'Edge Min Node Degree': avg_edge_min_node_degrees,
                    'Edge Rel Count': avg_edge_rel_counts,
                    'Edge Degree': avg_e_degs,
                    'N Entities': n_ent_network,
                    'N Relations': n_rel_network,
                    'N Triples': n_triples,
                    'N Connected Components': n_conn_comps,
                    'Median Relation Count': med_rel_count,
                    'Min Relation Count': min_rel_count,
                    'RE': rel_entropy,
                    'EE': ent_entropy}

    print(results_dict)

    # return results_dict, run_id, head_prediction_df, tail_prediction_df
    return results_dict, run_id
