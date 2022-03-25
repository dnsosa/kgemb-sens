# -*- coding: utf-8 -*-

"""Run the embedding pipeline."""

import pandas as pd

from pykeen.pipeline import pipeline
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df

from kgemb_sens.analyze.metrics import calc_edge_input_statistics, calc_network_input_statistics, calc_output_statistics
from kgemb_sens.transform.graph_utilities import undirect_multidigraph


def run_embed_pipeline(data_paths, i, params, train_conditions_id, G, test_edge, degree_dict=None, G_undir=None):
    if G_undir is None:
        G_undir = undirect_multidigraph(G)

    if len(data_paths) == 2:
        new_train_path, new_test_path = data_paths
    elif len(data_paths) == 3:
        new_train_path, new_val_path, new_test_path = data_paths

    result = pipeline(
        training=new_train_path,
        ##validation=new_val_path,
        testing=new_test_path,
        model=params["model_name"],
        # Training configuration
        training_kwargs=dict(
            num_epochs=params["n_epochs"],
            use_tqdm_batch=False,
        ),
        # Runtime configuration
        random_seed=1235,
        device='cpu',
    )

    run_id = f"{train_conditions_id}_model{params['model_name']}"

    test_triple = pd.read_csv(new_test_path, header=None, sep="\t").drop_duplicates()
    test_triple.columns = ["source", "edge", "target"]
    u, r, v = test_triple['source'][0], test_triple['edge'][0], test_triple['target'][0]

    edge_min_node_degree, edge_rel_count, e_deg = calc_edge_input_statistics(G, test_edge, degree_dict, G_undir=G_undir)
    net_stats = calc_network_input_statistics(G, calc_expensive=False, G_undir=G_undir)
    n_ent_network, n_rel_network, n_triples, n_conn_comps, med_rel_count, min_rel_count, rel_entropy, ent_entropy = net_stats

    head_prediction_df = get_head_prediction_df(result.model, str(r), str(v), triples_factory=result.training)
    tail_prediction_df = get_tail_prediction_df(result.model, str(u), str(r), triples_factory=result.training)

    head_deg_rank_corr = calc_output_statistics(list(head_prediction_df.head_label), degree_dict)
    tail_deg_rank_corr = calc_output_statistics(list(tail_prediction_df.tail_label), degree_dict)

    results_dict = {'Dataset': params["dataset"],
                    'PCNet_filter': params["pcnet_filter"],
                    'Model_name': params["model_name"],
                    'Sparsified_frac': params["sparsified_frac"],
                    "VT_alpha": params["vt_alpha"],
                    'Alpha': params["alpha"],
                    'Prob_type': params["prob_type"],
                    'Flatten_kg': params["flatten_kg"],
                    'Neg_Completion_Frac': params["neg_completion_frac"],
                    'Contradiction_Frac': params["contradiction_frac"],
                    'Contra_Remove_Frac': params["contra_remove_frac"],
                    'MODE': params["MODE"],
                    'Num_resamples': params["n_resample"],
                    # 'Val_test_subset_idx': str(val_test_subset_idx),
                    'Num_epochs': params["n_epochs"],
                    'Run': i,
                    'Run_ID': run_id,
                    'AMRI': result.metric_results.get_metric('adjusted_mean_rank_index'),
                    'Hits@1': result.metric_results.get_metric('hits@1'),
                    'Hits@3': result.metric_results.get_metric('hits@3'),
                    'Hits@5': result.metric_results.get_metric('hits@5'),
                    'Hits@10': result.metric_results.get_metric('hits@10'),
                    'MR': result.metric_results.get_metric('mean_rank'),
                    'MRR': result.metric_results.get_metric('mean_reciprocal_rank'),
                    'Head Deg Rank Corr': head_deg_rank_corr,
                    'Tail Deg Rank Corr': tail_deg_rank_corr,
                    'Edge Min Node Degree': edge_min_node_degree,
                    'Edge Rel Count': edge_rel_count,
                    'Edge Degree': e_deg,
                    'N Entities': n_ent_network,
                    'N Relations': n_rel_network,
                    'N Triples': n_triples,
                    'N Connected Components': n_conn_comps,
                    'Median Relation Count': med_rel_count,
                    'Min Relation Count:': min_rel_count,
                    'RE': rel_entropy,
                    'EE': ent_entropy}

    print(results_dict)

    return results_dict, run_id, head_prediction_df, tail_prediction_df
