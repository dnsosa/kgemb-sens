# -*- coding: utf-8 -*-

"""Run the embedding pipeline."""

import networkx as nx
import numpy as np
import pandas as pd

from collections import Counter

from pykeen.pipeline import pipeline
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df

from kgemb_sens.analyze.network_metrics import calc_edge_input_statistics, calc_network_input_statistics, calc_output_statistics
from kgemb_sens.transform.graph_utilities import undirect_multidigraph


def evaluate(G, task, setting, method, metric, n_neg_samples):
    """
    Evaluate KG embedding performance for input KGEmb.

    :param G: input KG
    :param task: prediction task (repurposing, target ID, disease-gene assoc?)
    :param setting: evaluation setting (T1: single edge reconstruct, T2: hidden test fraction)
    :param method: embedding method (RotatE, DistMult, etc.)
    :param metric: performance metric (MRR, AMRI, etc.)
    :param n_neg_samples: number of negative samples
    """
    pass  # This is now all taken care of in the CLI.


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
        optimizer_kwargs=dict(
            lr=params["learning_rate"]
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

    # TODO: In the corruption regimes, think about what is considered a test triple and if/how that changes when rel types get shuffled around
    #print("Calculating test edge statistics...")
    #head_ranks = []
    #num_valid_heads = []
    #tail_ranks = []
    #num_valid_tails = []
    #us, vs, rs = [], [], []
    #edge_min_node_degrees, edge_rel_counts, e_degs = [], [], []
    #test_edge_counter = 0
    #G_rel_set = [rel for _, _, rel in G.edges(data="edge")]
    #G_rel_counter = Counter(G_rel_set)
    #for test_edge in test_subset:
    #    # print(f"Test edge: {test_edge}")
    #    u, v, r = test_edge[0], test_edge[1], test_edge[-1]['edge']
    #    us.append(u)
    #    vs.append(v)
    #    rs.append(r)
    #
    #    head_prediction_df = get_head_prediction_df(result.model, str(r), str(v), triples_factory=result.training)
    #    valid_drug_preds = [label for label in head_prediction_df.head_label if 'Compound' in label]
    #    head_ranks.append(valid_drug_preds.index(u))
    #    num_valid_heads.append(len(valid_drug_preds))
    #
    #    tail_prediction_df = get_tail_prediction_df(result.model, str(u), str(r), triples_factory=result.training)
    #    valid_disease_preds = [label for label in tail_prediction_df.tail_label if 'Disease' in label]
    #    tail_ranks.append(valid_disease_preds.index(v))
    #    num_valid_tails.append(len(valid_disease_preds))
    #
    #    edge_min_node_degree, edge_rel_count, e_deg = calc_edge_input_statistics(G, test_edge, degree_dict,
    #                                                                             G_undir=G_undir,
    #                                                                             G_rel_counter=G_rel_counter)
    #    edge_min_node_degrees.append(edge_min_node_degree)
    #    edge_rel_counts.append(edge_rel_count)
    #    e_degs.append(e_deg)
    #    test_edge_counter += 1
    #    if test_edge_counter % 100 == 0:
    #        print(f"Processed {test_edge_counter} test edges")
    #
    #test_ranks_df = pd.DataFrame()
    #test_ranks_df["head_ranks"] = head_ranks
    #test_ranks_df["num_valid_heads"] = num_valid_heads
    #test_ranks_df["tail_ranks"] = tail_ranks
    #test_ranks_df["num_valid_tails"] = num_valid_tails
    #test_ranks_df["source"] = us
    #test_ranks_df["edge"] = rs
    #test_ranks_df["target"] = vs
    #test_ranks_df["e_mnd"] = edge_min_node_degrees
    #test_ranks_df["e_rel_counts"] = edge_rel_counts
    #test_ranks_df["e_degs"] = e_degs
    #
    ## print(f"edge_min_node_degrees: {edge_min_node_degrees}")
    #avg_edge_min_node_degrees = np.average(edge_min_node_degrees)
    #
    ## print(f"edge_rel_counts: {edge_rel_counts}")
    #avg_edge_rel_counts = np.average(edge_rel_counts)
    #
    ## print(f"e_degs: {e_degs}")
    #avg_e_degs = np.average(e_degs)
    #print("Test edge statistics done.")

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
                    'Topo_perturb_method': params["topo_perturb_method"],
                    'Topo_perturb_strength': params["topo_perturb_strength"],
                    'Rel_corrupt_method': params["rel_corrupt_method"],
                    'Rel_corrupt_strength': params["rel_corrupt_strength"],
                    'Eval_task': params["eval_task"],
                    'Eval_setting': params["eval_setting"],
                    'VT_alpha': params["vt_alpha"],
                    'Alpha': params["alpha"],
                    'Prob_type': params["prob_type"],
                    'Flatten_kg': params["flatten_kg"],
                    'randomize_relations': params["randomize_relations"],
                    'single_relation': params["single_relation"],
                    'hub_remove_thresh': params["hub_remove_thresh"],
                    'test_min_edeg': params["test_min_edeg"],
                    'test_max_edeg': params["test_max_edeg"],
                    'test_min_mnd': params["test_min_mnd"],
                    'test_max_mnd': params["test_max_mnd"],
                    'replace_edges': params["replace_edges"],
                    'Num_resamples': params["n_resample"],
                    'Num_negative_samples': params["n_negatives"],
                    'Num_epochs': params["n_epochs"],
                    'Learning_rate': params["learning_rate"],
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
                    #'Edge Min Node Degree': avg_edge_min_node_degrees,
                    #'Edge Rel Count': avg_edge_rel_counts,
                    #'Edge Degree': avg_e_degs,
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
    #return results_dict, run_id, test_ranks_df
    return results_dict, run_id
