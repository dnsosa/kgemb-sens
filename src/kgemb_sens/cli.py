# -*- coding: utf-8 -*-

"""Command line interface for kgemb_sens."""

import os

import click
import networkx as nx
import numpy as np
import pandas as pd

from kgemb_sens.analyze.embed import run_embed_pipeline
from kgemb_sens.analyze.metrics import calc_network_input_statistics
from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts, load_drkg_data
from kgemb_sens.transform.contradiction_utilities import find_all_valid_negations
from kgemb_sens.transform.graph_utilities import undirect_multidigraph
from kgemb_sens.transform.processing_pipeline import graph_processing_pipeline

DATA_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/pykeen/datasets"

# TODO Fill this all in
@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--data_dir', 'data_dir', default=DATA_DIR)
@click.option('--dataset', 'dataset', default='nations')
@click.option('--pcnet_filter/--no-pcnet_filter', 'pcnet_filter', default=False)
@click.option('--pcnet_dir', 'pcnet_dir', default=DATA_DIR)
@click.option('--val_test_frac', 'val_test_frac', default=1.0)
@click.option('--val_frac', 'val_frac', default=0.0)
@click.option('--sparsified_frac', 'sparsified_frac', default=0.0)
@click.option('--alpha', 'alpha', default=0.0)
@click.option('--n_resample', 'n_resample', default=100)
@click.option('--prob_type', 'prob_type', default='degree')
@click.option('--flatten_kg', 'flatten_kg', default=False)
@click.option('--neg_completion_frac', 'neg_completion_frac', default=0.0)
@click.option('--contradiction_frac', 'contradiction_frac', default=0.0)
@click.option('--contra_remove_frac', 'contra_remove_frac', default=0.0)
@click.option('--MODE', 'MODE', default='contrasparsify')
@click.option('--model_name', 'model_name', default='transe')
@click.option('--neg_completion_frac', 'neg_completion_frac', default=0.0)
@click.option('--n_epochs', 'n_epochs', default=200)
def main(out_dir, data_dir, dataset, pcnet_filter, pcnet_dir, val_test_frac, val_frac, sparsified_frac, alpha, n_resample, prob_type, flatten_kg,
         neg_completion_frac, contradiction_frac, contra_remove_frac, MODE, model_name, n_epochs):
    """Run main function."""

    SEED = 1005
    np.random.seed(SEED)

    os.makedirs(out_dir, exist_ok=True)

    params = {"dataset": dataset,
              "pcnet_filter": pcnet_filter,
              "val_test_frac": val_test_frac,
              "val_frac": val_frac,
              "sparsified_frac": sparsified_frac,
              "alpha": alpha,
              "n_resample": n_resample,
              "prob_type": prob_type,  # distance, degree
              "flatten_kg": flatten_kg,
              "neg_completion_frac": neg_completion_frac, #TODO: Negative completion fraction is silly. What about multiple of other edges. NEED better sampling strategy I think. Randomized algorithm?
              "contradiction_frac": contradiction_frac,
              "contra_remove_frac": contra_remove_frac,
              "MODE": MODE,  # "sparsification", "contradictification", "contrasparsify"
              "model_name": model_name,
              "n_epochs": n_epochs}

    # LOAD DATA
    print("Loading data...")
    if dataset in ["nations", "umls", "countries", "kinships"]:
        G = load_benchmark_data_three_parts(dataset, data_dir)
    elif dataset in ["gnbr_gg", "gnbr_drdz", "gnbr_drg", "drugbank_drdz", "drugbank_drg", "string_gg"]:
        G = load_drkg_data(dataset, data_dir, pcnet_filter, pcnet_dir)
    print("\n\nData load and network creation complete.\n\n")
    all_valid_negations = []
    all_rels = set([r for _, _, r in G.edges(data='edge')])

    # TODO: Simplify to not include contradictification option (contra_remove_frac = 0)
    if (MODE in ["contradictification", "contrasparsify"]) and (neg_completion_frac > 0):
        print("\n\nFinding all valid negations\n\n")
        all_valid_negations = find_all_valid_negations(G)
        print("All valid negations found.")

    all_results_list = []

    # Precompute distance matrix if requested because it's expensive
    if (prob_type == "distance") and (alpha != 0):
        dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(undirect_multidigraph(G))))
    else:
        dist_mat = None

    # Precompute degree dictionary
    if flatten_kg:
        G_flat = nx.Graph(undirect_multidigraph(G))
        degree_dict = dict(G_flat.degree())
    else:
        degree_dict = dict(G.degree())

    print("\n\nBeginning graph processing pipeline...")
    for i in range(n_resample):
        print(f"Sample {i}\n")
        data_paths, train_conditions_id, edge_divisions, G_con = graph_processing_pipeline(G, i, params, out_dir,
                                                                                           all_valid_negations,
                                                                                           all_rels, SEED,
                                                                                           dist_mat=dist_mat,
                                                                                           degree_dict=degree_dict)
        train_subset, test_subset, sparse_subset, new_contradictions, removed_contradictions = edge_divisions
        print("Now embedding results...")
        results_dict, run_id, head_pred_df, tail_pred_df = run_embed_pipeline(data_paths, i, params,
                                                                              train_conditions_id,
                                                                              G, test_subset[0],
                                                                              degree_dict)

        # TODO: output embeddings from training

        # TODO: Doesn't make sense to keep reassigning this every loop. Create the run ID sooner
        # Make save directory
        save_dir = f"{out_dir}/results/{run_id}/"  # this directory
        os.makedirs(save_dir, exist_ok=True)

        head_pred_df.to_csv(f"{save_dir}/head_pred_run_{i}.tsv", sep='\t', header=True, index=False)
        tail_pred_df.to_csv(f"{save_dir}/tail_pred_run_{i}.tsv", sep='\t', header=True, index=False)

        print("\nDone embedding.\n")
        all_results_list.append(results_dict)

    print("\nFinished with all resamples!")

    print(f"Saving all results and metrics to: {save_dir}")

    # Calculate network stats
    calc_diam = (G.number_of_edges() < 100)
    network_stats_results = calc_network_input_statistics(G, calc_diam)
    network_stats_dict = {"n_ent_network": network_stats_results[0],
                          "n_rel_network": network_stats_results[1],
                          "n_conn_comps": network_stats_results[2],
                          "avg_cc": network_stats_results[3],
                          "med_rel_count": network_stats_results[4],
                          "min_rel_count": network_stats_results[5]}
    if calc_diam:
        network_stats_dict["diam"] = network_stats_results[6]

    network_stats_df = pd.DataFrame(network_stats_dict, index=[0])
    network_stats_df.to_csv(f"{save_dir}/network_stats.tsv", sep='\t', header=True, index=False)

    # plot_graph_nice(GTest,
    #                 train_subset,
    #                 test_subset,
    #                 sparse_subset=sparse_subset,
    #                 new_contradictions=new_contradictions,
    #                 removed_contradictions=removed_contradictions)

    res_df = pd.DataFrame(all_results_list)
    res_df.to_csv(f"{save_dir}/results.df", sep='\t', header=True, index=False)


if __name__ == '__main__':
    main()



#python -m kgemb_sens --output_folder /Users/dnsosa/Desktop/AltmanLab/KGEmbSensitivity/test_out --data_dir /Users/dnsosa/.data/pykeen/datasets --n_resample 1 --n_epochs 2
