# -*- coding: utf-8 -*-

"""Command line interface for kgemb_sens."""

import os

import click
import networkx as nx
import numpy as np
import pandas as pd

from kgemb_sens.analyze.eval_embed_performance import run_embed_pipeline
from kgemb_sens.analyze.network_metrics import calc_network_input_statistics
from kgemb_sens.experiments.network_perturb import add_self_loops, remove_hubs, upsample_low_deg_triples, degree_based_downsample
from kgemb_sens.experiments.relation_corruptions import perturb_relations
from kgemb_sens.load.network_load import load_benchmark_data_three_parts, load_drkg_data, load_covid_graph
from kgemb_sens.transform.graph_utilities import make_all_one_type, preprocess_remove_hubs, randomize_edges, undirect_multidigraph
from kgemb_sens.transform.processing_pipeline import simplified_graph_processing_pipeline
from kgemb_sens.utilities import retrieve_rel_whitelist

DATA_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/pykeen/datasets"
#DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"
COVIDKG_DIR = "/oak/stanford/groups/rbaltman/dnsosa/KGEmbSensitivity/covid19kg"


# TODO: Update this

@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--data_dir', 'data_dir', default=DATA_DIR)
@click.option('--dataset', 'dataset', default='nations')
@click.option('--pcnet_filter/--no-pcnet_filter', 'pcnet_filter', default=False)
@click.option('--pcnet_dir', 'pcnet_dir', default=DATA_DIR)
@click.option('--covidkg_dir', 'covidkg_dir', default=COVIDKG_DIR)
@click.option('--randomize_relations/--no-randomize_relations', 'randomize_relations', default=False)
@click.option('--single_relation/--no-single_relation', 'single_relation', default=False)
@click.option('--hub_remove_thresh', 'hub_remove_thresh', default=float("inf"))
@click.option('--topo_perturb_method', 'topo_perturb_method', default="self_loops")
@click.option('--topo_perturb_strength', 'topo_perturb_strength', default=1.0)
@click.option('--rel_corruption_method', 'rel_corrupt_method', default="corrupt")
@click.option('--rel_corruption_strength', 'rel_corrupt_strength', default=1.0)
@click.option('--eval_setting', 'eval_setting', default="single_edge")
@click.option('--eval_task', 'eval_task', default='DrDz')
@click.option('--val_test_frac', 'val_test_frac', default=None)
@click.option('--val_frac', 'val_frac', default=0.0)  # CHANGE THIS!
@click.option('--vt_alpha', 'vt_alpha', default=0.0)
@click.option('--test_min_edeg', 'test_min_edeg', default=0.0)
@click.option('--test_max_edeg', 'test_max_edeg', default=float("inf"))
@click.option('--test_min_mnd', 'test_min_mnd', default=0.0)
@click.option('--test_max_mnd', 'test_max_mnd', default=float("inf"))
@click.option('--sparsified_frac', 'sparsified_frac', default=0.0)
@click.option('--alpha', 'alpha', default=1.0)
@click.option('--n_resample', 'n_resample', default=100)  # TODO: What's the deal with this?
@click.option('--n_negatives', 'n_negatives', default=1)
@click.option('--prob_type', 'prob_type', default='degree')
@click.option('--flatten_kg', 'flatten_kg', default=False)
@click.option('--replace_edges/--no-replace_edges', 'replace_edges', default=True)
@click.option('--model_name', 'model_name', default='transe')
@click.option('--n_epochs', 'n_epochs', default=200)
@click.option('--learning_rate', 'learning_rate', default=0.02)
def main(out_dir, data_dir, dataset, pcnet_filter, pcnet_dir, covidkg_dir, randomize_relations, single_relation,
         hub_remove_thresh, topo_perturb_method, topo_perturb_strength, rel_corrupt_method, rel_corrupt_strength,
         eval_setting, eval_task, val_frac, val_test_frac, vt_alpha, test_min_edeg, test_max_edeg,
         test_min_mnd, test_max_mnd, sparsified_frac, alpha, n_resample, n_negatives, prob_type, flatten_kg, replace_edges,
         model_name, n_epochs, learning_rate):
    """Run main function."""

    SEED = 1005
    np.random.seed(SEED)

    print("In the CLI Main function. Seed is set. Imports are imported.")
    os.makedirs(out_dir, exist_ok=True)

    if val_test_frac is None:
        val_test_frac = 1.0 if eval_setting == "single_edge" else 0.9

    params = {"dataset": dataset,
              "pcnet_filter": pcnet_filter,
              "randomize_relations": randomize_relations,
              "single_relation": single_relation,
              "hub_remove_thresh": hub_remove_thresh,
              "topo_perturb_method": topo_perturb_method,
              "topo_perturb_strength": topo_perturb_strength,
              "rel_corrupt_method": rel_corrupt_method,
              "rel_corrupt_strength": rel_corrupt_strength,
              "eval_setting": eval_setting,
              "eval_task": eval_task,
              "val_frac": val_frac,
              "val_test_frac": val_test_frac,
              "vt_alpha": vt_alpha,
              "test_min_edeg": test_min_edeg,
              "test_max_edeg": test_max_edeg,
              "test_min_mnd": test_min_mnd,
              "test_max_mnd": test_max_mnd,
              "sparsified_frac": sparsified_frac,
              "alpha": alpha,
              "n_resample": n_resample,
              "n_negatives": n_negatives,
              "prob_type": prob_type,  # distance, degree
              "flatten_kg": flatten_kg,
              "replace_edges": replace_edges,
              "model_name": model_name,
              "n_epochs": n_epochs,
              "learning_rate": learning_rate}

    print("Running on these params:")
    print(params)

    # LOAD DATA
    antonyms = None
    print("Loading data...")
    if dataset in ["nations", "umls", "countries", "kinships"]:
        G = load_benchmark_data_three_parts(dataset, data_dir)
    elif dataset in ["gnbr_gg", "gnbr_drdz", "gnbr_drg", "drugbank_drdz", "drugbank_drg", "string_gg", "gnbr", "hetionet"]:
        G = load_drkg_data(dataset, data_dir, pcnet_filter, pcnet_dir)
    elif dataset == "covid":
        G = load_covid_graph(covidkg_dir)

    # Calculate the relevant whitelist if any
    rel_whitelist = retrieve_rel_whitelist(dataset, eval_task)

    # Optional pre-processing as controls/debugging
    dr_dz_whitelist_pairs = None  # only used in the single relation condition
    if hub_remove_thresh is not None:
        G = preprocess_remove_hubs(G, hub_size=hub_remove_thresh)
    if randomize_relations:
        G = randomize_edges(G, rel_whitelist)
    if single_relation:
        G, dr_dz_whitelist_pairs = make_all_one_type(G, rel_whitelist)

    # Pre-processing corruption experiments:
    # First alter the topology
    if topo_perturb_method == "self_loops":
        G = add_self_loops(G, fill_frac=topo_perturb_strength, SEED=SEED)
    elif topo_perturb_method == "remove_hubs":
        G = remove_hubs(G, frac_nodes=topo_perturb_strength, SEED=SEED)  # TODO: implement
    elif topo_perturb_method == "upsample_low_deg":
        G = upsample_low_deg_triples(G, frac_triples=topo_perturb_strength, SEED=SEED)  # TODO: implement
    elif topo_perturb_method == "degree_based_downsample":
        G = degree_based_downsample(G, frac_triples=topo_perturb_strength, SEED=SEED)  # TODO: implement
    else:
        print(f"{topo_perturb_method} is an invalid topology perturbing method!")
        return None

    # Used for the distance matrix calculations and in the flattened case
    G_undir = undirect_multidigraph(G)

    all_results_list = []
    all_rels = set([r for _, _, r in G.edges(data='edge')])

    print("\nData load and network creation complete.")

    # Then corrupt the relations
    if rel_corrupt_method in {"corrupt", "flatten"}:
        G = perturb_relations(G, rel_corrupt_strength,
                              whitelist_rels=rel_whitelist,
                              all_possible_rels=list(all_rels),
                              perturb_method=rel_corrupt_method,
                              SEED=SEED)
    else:
        print(f"{rel_corrupt_method} is an invalid relation corruption method!")
        return None

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

    # TODO: Need to double check this graph processing pipeline still working for me....
    # TODO: What can be gutted for different kinds of topo corruptors.
    print("\n\nBeginning graph processing pipeline...")
    for i in range(n_resample):
        print(f"\nSample {i}")

        data_paths, train_conditions_id, edge_divisions, G_out = \
            simplified_graph_processing_pipeline(G, i, params, out_dir, SEED, G_undir=G_undir,
                                                 dist_mat=dist_mat, degree_dict=degree_dict,
                                                 test_min_edeg=test_min_edeg,
                                                 test_max_edeg=test_max_edeg, test_min_mnd=test_min_mnd,
                                                 test_max_mnd=test_max_mnd, rel_whitelist=rel_whitelist,
                                                 dr_dz_whitelist_pairs=dr_dz_whitelist_pairs)

        G_out_undir = undirect_multidigraph(G_out)
        G_out_degree_dict = dict(G_out.degree())

        train_subset, test_subset = edge_divisions
        print("Now embedding results...")

        # results_dict, run_id, head_pred_df, tail_pred_df = run_embed_pipeline(data_paths, i, params,
        results_dict, run_id, test_ranks_df = run_embed_pipeline(data_paths, i, params, train_conditions_id,
                                                                 G_out, test_subset, degree_dict=G_out_degree_dict,
                                                                 G_undir=G_out_undir, rel_whitelist=rel_whitelist)


        # TODO: output embeddings from training
        # TODO: Doesn't make sense to keep reassigning this every loop. Create the run ID sooner
        # Make save directory
        save_dir = f"{out_dir}/results/{run_id}/"  # this directory
        os.makedirs(save_dir, exist_ok=True)

        # head_pred_df.to_csv(f"{save_dir}/head_pred_run_{i}.tsv", sep='\t', header=True, index=False)
        # tail_pred_df.to_csv(f"{save_dir}/tail_pred_run_{i}.tsv", sep='\t', header=True, index=False)
        test_ranks_df.to_csv(f"{save_dir}/test_ranks_df_run_{i}.tsv", sep='\t', header=True, index=False)

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
