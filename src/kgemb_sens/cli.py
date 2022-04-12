# -*- coding: utf-8 -*-

"""Command line interface for kgemb_sens."""

import os

import click
import itertools
import networkx as nx
import numpy as np
import pandas as pd

from collections import Counter

from kgemb_sens.analyze.embed import run_embed_pipeline
from kgemb_sens.analyze.metrics import calc_network_input_statistics
from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts, load_drkg_data, load_covid_graph
from kgemb_sens.transform.contradiction_utilities import find_all_valid_negations
from kgemb_sens.transform.graph_utilities import undirect_multidigraph, remove_E, filter_in_etype, randomize_edges, make_all_one_type, remove_hubs
from kgemb_sens.transform.processing_pipeline import graph_processing_pipeline

DATA_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/pykeen/datasets"
COVIDKG_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/covid19kg"


@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--data_dir', 'data_dir', default=DATA_DIR)
@click.option('--dataset', 'dataset', default='nations')
@click.option('--pcnet_filter/--no-pcnet_filter', 'pcnet_filter', default=False)
@click.option('--pcnet_dir', 'pcnet_dir', default=DATA_DIR)
@click.option('--covidkg_dir', 'covidkg_dir', default=COVIDKG_DIR)
@click.option('--dengue_filter/--no-dengue_filter', 'dengue_filter', default=False)
@click.option('--dengue_expand_depth', 'dengue_expand_depth', default=1)
@click.option('--remove_E/--no-remove_E', 'remove_E_filter', default=False)
@click.option('--filter_in_antonyms/--no-filter_in_antonyms', 'filter_in_antonyms', default=False)
@click.option('--randomize_relations/--no-randomize_relations', 'randomize_relations', default=False)
@click.option('--single_relation/--no-single_relation', 'single_relation', default=False)
@click.option('--hub_remove_thresh', 'hub_remove_thresh', default=float("inf"))
@click.option('--val_test_frac', 'val_test_frac', default=1.0)
@click.option('--val_frac', 'val_frac', default=0.0)
@click.option('--vt_alpha', 'vt_alpha', default=0.0)
@click.option('--test_min_edeg', 'test_min_edeg', default=0.0)
@click.option('--test_max_edeg', 'test_max_edeg', default=float("inf"))
@click.option('--test_min_mnd', 'test_min_mnd', default=0.0)
@click.option('--test_max_mnd', 'test_max_mnd', default=float("inf"))
@click.option('--sparsified_frac', 'sparsified_frac', default=0.0)
@click.option('--alpha', 'alpha', default=0.0)
@click.option('--n_resample', 'n_resample', default=100)
@click.option('--prob_type', 'prob_type', default='degree')
@click.option('--flatten_kg', 'flatten_kg', default=False)
@click.option('--neg_completion_frac', 'neg_completion_frac', default=0.0)
@click.option('--contradiction_frac', 'contradiction_frac', default=0.0)
@click.option('--contra_remove_frac', 'contra_remove_frac', default=0.0)
@click.option('--replace_edges/--no-replace_edges', 'replace_edges', default=True)
@click.option('--MODE', 'MODE', default='contrasparsify')
@click.option('--model_name', 'model_name', default='transe')
@click.option('--n_epochs', 'n_epochs', default=200)
@click.option('--repurposing_eval/--no-repurposing_eval', 'repurposing_evaluation', default=False)
def main(out_dir, data_dir, dataset, pcnet_filter, pcnet_dir, covidkg_dir, dengue_filter, dengue_expand_depth,
         remove_E_filter, filter_in_antonyms, randomize_relations, single_relation, hub_remove_thresh,
         val_test_frac, val_frac, vt_alpha, test_min_edeg, test_max_edeg, test_min_mnd, test_max_mnd,
         sparsified_frac, alpha, n_resample, prob_type, flatten_kg, neg_completion_frac,
         contradiction_frac, contra_remove_frac, replace_edges, MODE, model_name, n_epochs, repurposing_evaluation):
    """Run main function."""

    SEED = 1005
    np.random.seed(SEED)

    print("In the CLI Main function. Seed is set. Imports are imported.")
    os.makedirs(out_dir, exist_ok=True)

    params = {"dataset": dataset,
              "pcnet_filter": pcnet_filter,
              "remove_E_filter": remove_E_filter,
              "filter_in_antonyms": filter_in_antonyms,
              "randomize_relations": randomize_relations,
              "single_relation": single_relation,
              "hub_remove_thresh": hub_remove_thresh,
              "val_test_frac": val_test_frac,
              "val_frac": val_frac,
              "vt_alpha": vt_alpha,
              "test_min_edeg": test_min_edeg,
              "test_max_edeg": test_max_edeg,
              "test_min_mnd": test_min_mnd,
              "test_max_mnd": test_max_mnd,
              "sparsified_frac": sparsified_frac,
              "alpha": alpha,
              "n_resample": n_resample,
              "prob_type": prob_type,  # distance, degree
              "flatten_kg": flatten_kg,
              "neg_completion_frac": neg_completion_frac, #TODO: NEED better sampling strategy I think. Randomized algorithm?
              "contradiction_frac": contradiction_frac,
              "contra_remove_frac": contra_remove_frac,
              "replace_edges": replace_edges,
              "MODE": MODE,  # "sparsification", "contrasparsify"
              "model_name": model_name,
              "n_epochs": n_epochs,
              "repurposing_evaluation": repurposing_evaluation}

    print("Running on these params:")
    print(params)

    # LOAD DATA
    antonyms = None
    print("Loading data...")
    if dataset in ["nations", "umls", "countries", "kinships"]:
        G = load_benchmark_data_three_parts(dataset, data_dir)
    elif dataset in ["gnbr_gg", "gnbr_drdz", "gnbr_drg", "drugbank_drdz", "drugbank_drg", "string_gg"]:
        G = load_drkg_data(dataset, data_dir, pcnet_filter, pcnet_dir, dengue_filter, dengue_expand_depth)
        # Declare the dataset's antonym pairs. TODO: Should this be somewhere else?
        if dataset == "gnbr_drg":
            antonyms = [("E+", "E-"), ("A+", "A-")]
        elif dataset == "gnbr_drdz":
            antonyms = [("T", "J")]
    elif dataset == "covid":
        G = load_covid_graph(covidkg_dir)
        antonyms = [('increases', 'decreases'), ('positiveCorrelation', 'negativeCorrelation')]

    # Optional pre-processing as controls/debugging
    if hub_remove_thresh is not None:
        G = remove_hubs(G, hub_size=hub_remove_thresh)
    if remove_E_filter:
        G = remove_E(G)
    if filter_in_antonyms:
        G = filter_in_etype(G, list(itertools.chain(*antonyms)))
    if randomize_relations:
        G = randomize_edges(G)
    if single_relation:
        G = make_all_one_type(G)

    G_undir = undirect_multidigraph(G)

    print("\nData load and network creation complete.")
    all_valid_negations = []
    all_rels = set([r for _, _, r in G.edges(data='edge')])

    if (MODE == "contrasparsify") and (neg_completion_frac > 0):
        print("\n\nFinding all valid negations\n\n")
        all_valid_negations = find_all_valid_negations(G)
        print("All valid negations found.")

    all_results_list = []

    # Precompute distance matrix if requested because it's expensive
    if (prob_type == "distance") and (alpha != 0):
        dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(G_undir)))
    else:
        dist_mat = None

    # Precompute degree dictionary
    if flatten_kg:
        G_flat = nx.Graph(G_undir)
        degree_dict = dict(G_flat.degree())
    else:
        degree_dict = dict(G.degree())

    # Calculate the relevant whitelist if any
    rel_whitelist = None
    if repurposing_evaluation:
        if "gnbr" in dataset:
            rel_whitelist = {"T", "Pa"}
        elif dataset == "hetionet":
            rel_whitelist = {"CtD", "CpD"}

    print("\n\nBeginning graph processing pipeline...")
    for i in range(n_resample):
        print(f"\nSample {i}")
        data_paths, train_conditions_id, edge_divisions, G_out = graph_processing_pipeline(G, i, params, out_dir,
                                                                                           all_valid_negations,
                                                                                           all_rels, SEED,
                                                                                           G_undir=G_undir,
                                                                                           antonyms=antonyms,
                                                                                           dist_mat=dist_mat,
                                                                                           degree_dict=degree_dict,
                                                                                           replace_edges=replace_edges,
                                                                                           test_min_edeg=test_min_edeg,
                                                                                           test_max_edeg=test_max_edeg,
                                                                                           test_min_mnd=test_min_mnd,
                                                                                           test_max_mnd=test_max_mnd,
                                                                                           rel_whitelist=rel_whitelist)

        G_out_undir = undirect_multidigraph(G_out)
        G_out_degree_dict = dict(G_out.degree())

        train_subset, test_subset, sparse_subset, new_contradictions, removed_contradictions = edge_divisions
        print("Now embedding results...")

        results_dict, run_id, head_pred_df, tail_pred_df = run_embed_pipeline(data_paths, i, params,
                                                                              train_conditions_id,
                                                                              G_out, test_subset[0],
                                                                              G_out_degree_dict,
                                                                              G_undir=G_out_undir,
                                                                              rel_whitelist=rel_whitelist)

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

    # print(f"\mSaving all results and metrics to: {save_dir}")

    # Calculate network stats
    # Input network
    calc_expensive = (G.number_of_edges() < 500)
    network_stats_results = calc_network_input_statistics(G, calc_expensive, G_undir=G_undir)
    network_stats_dict = {"n_ent_network": network_stats_results[0],
                          "n_rel_network": network_stats_results[1],
                          "n_triples": network_stats_results[2],
                          "n_conn_comps": network_stats_results[3],
                          "med_rel_count": network_stats_results[4],
                          "min_rel_count": network_stats_results[5],
                          "RE": network_stats_results[6],
                          "EE": network_stats_results[7]}

    # Network post modification
    # network_stats_results_post = calc_network_input_statistics(G_out, calc_expensive, G_undir=G_undir)
    # network_stats_dict_post = {"post_n_ent_network": network_stats_results_post[0],
    #                            "post_n_rel_network": network_stats_results_post[1],
    #                            "post_n_triples": network_stats_results_post[2],
    #                            "post_n_conn_comps": network_stats_results_post[3],
    #                            "post_med_rel_count": network_stats_results_post[4],
    #                            "post_min_rel_count": network_stats_results_post[5]}

    if calc_expensive:
        network_stats_dict["avg_cc"] = network_stats_results[8]
        # network_stats_dict_post["post_avg_cc"] = network_stats_results_post[8]
        network_stats_dict["diam"] = network_stats_results[9]
        # network_stats_dict_post["diam"] = network_stats_results_post[9]

    # network_stats_dict.update(network_stats_dict_post)

    network_stats_df = pd.DataFrame(network_stats_dict, index=[0])
    network_stats_df.to_csv(f"{save_dir}/network_stats.tsv", sep='\t', header=True, index=False)

    res_df = pd.DataFrame(all_results_list)
    res_df.to_csv(f"{save_dir}/results.df", sep='\t', header=True, index=False)


if __name__ == '__main__':
    main()



#python -m kgemb_sens --output_folder /Users/dnsosa/Desktop/AltmanLab/KGEmbSensitivity/test_out --data_dir /Users/dnsosa/.data/pykeen/datasets --n_resample 1 --n_epochs 2
