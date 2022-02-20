# -*- coding: utf-8 -*-

"""Run the embedding pipeline."""

import pykeen
from pykeen.pipeline import pipeline


def run_embed_pipeline(data_paths, i, params, train_conditions_id):
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

    model = result.model
    # tf = model.triples_factory
    run_id = f"{train_conditions_id}_model{params['model_name']}"

    results_dict = {'Dataset': params["dataset"],
                    'PCNet_filter': params["pcnet_filter"],
                    'Model_name': params["model_name"],
                    'Sparsified_frac': params["sparsified_frac"],
                    'Alpha': params["alpha"],
                    'Prob_type': params["prob_type"],
                    'Neg_Completion_Frac': params["neg_completion_frac"],
                    'Contradiction_Frac': params["contradiction_frac"],
                    'Contra_Remove_Frac': params["contra_remove_frac"],
                    'MODE': params["MODE"],
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
                    'MRR': result.metric_results.get_metric('mean_reciprocal_rank')}

    return results_dict, run_id
