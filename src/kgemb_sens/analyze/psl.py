# -*- coding: utf-8 -*-

"""Run the embedding pipeline."""

import itertools
import os

import numpy as np
import pandas as pd

from pykeen.pipeline import pipeline
from pykeen.models.predict import get_tail_prediction_df, get_head_prediction_df

from kgemb_sens.analyze.metrics import calc_edge_input_statistics, calc_network_input_statistics, calc_output_statistics
from kgemb_sens.transform.graph_utilities import undirect_multidigraph

# TODO: where should psl_dir be?


def run_psl_pipeline(data_paths, i, params, train_conditions_id, G, test_edges, train_edges, dataset="gnbr",
                       degree_dict=None, G_undir=None, psl_dir=PSL_DIR):

    create_data_files_for_psl(train_edges, test_edges, psl_dir, int(params["num_negatives"]), params["full_product"])

    # Now create a PSL rules file
    create_psl_rules_file()

    # Then run the thing....

    # And calculate all the metrics...


def create_psl_rules_file():
    pass


def create_data_files_for_psl(train_edges, test_edges, psl_dir, num_negatives, full_product):

    # Prepare mappings and general compatibilities with PSL expectations
    os.makedirs(psl_dir, exist_ok=True)
    edge_name_map = {"A+": "Ap",
                     "A-": "An",
                     "E+": "Ep",
                     "E-": "En",
                     "V+": "Vp",
                     "Gr>G": "GrrG"}

    def map_edge_name(in_edge_name, edge_name_map=edge_name_map):
        if in_edge_name in edge_name_map:
            return edge_name_map[edge_name]
        else:
            return edge_name

    # Need to map nodes to ints?
    node_map = dict(zip(list(G.nodes()), np.arange(len(G.nodes()))))  # TODO: is this necessary?

    # Create the data directories that PSL package is expecting
    # NOTE: Make sure this is just training triples!
    # edge_names = set([r for _, _, r in G.edges(data='edge')])
    edge_names = set([r['edge'] for _, _, _, r in train_edges])
    for edge_name in edge_names:
        relevant_edges = [e for e in train_edges if e[-1]['edge'] == edge_name]

        # Negative sampling
        relevant_edges_node_tuples = set([(u, v) for u, v, _, _ in relevant_edges])
        all_nodes = set(list(itertools.chain(*relevant_edges_node_tuples)))
        # head_nodes = set([u for u, _ in relevant_edges_node_tuples])
        # tail_nodes = set([v for _, v in relevant_edges_node_tuples])
        # NOTE: Should think about if when I corrupt head, I only resample other head nodes (i.e. just drugs),
        # or any node, i.e. could be a disease

        # Repeat procedure until have enough negative samples
        negative_edges = set([])
        while len(negative_edges) < (len(relevant_edges) * num_negatives):
            sampled_edge = np.random.choice(relevant_edges)
            if np.random.binomial(1, .5):  # flip a coin
                # head corrupt
                corrupt_edge = (np.random.choice(all_nodes), sampled_edge[1])
            else:
                # tail corrupt
                corrupt_edge = (sampled_edge[0], np.random.choice(all_nodes))

            if (corrupt_edge not in relevant_edges_node_tuples) and (corrupt_edge not in negative_edges):
                negative_edges.add(corrupt_edge)

        # Write edges
        edge_name_mapped = map_edge_name(edge_name)
        with open(f'{psl_dir}/{edge_name_mapped}_obs.txt', 'w') as fp:
            # Positive examples
            fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v, _, _ in relevant_edges))
            # Negative examples
            fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t0' for u, v in negative_edges))

    # Now make the target and truth files...
    # Target file
    test_edge_names = set([r['edge'] for _, _, _, r in test_edges])
    for test_edge_name in test_edge_names:

        relevant_test_edges_tuples = set([(u, v) for u, v, _, _ in relevant_test_edges])

        # Take the product of the head nodes for the edge type by the tail nodes for the edge type
        if full_product:
            relevant_train_edges = [e for e in train_edges if e[-1]['edge'] == test_edge_name]
            relevant_train_edges_tuples = set([(u, v) for u, v, _, _ in relevant_train_edges])
            train_edge_heads = [u for u, _ in relevant_train_edges]
            train_edge_tails = [v for _, v in relevant_train_edges]
            relevant_product_edges = set(list(itertools.product(train_edge_heads, train_edge_tails)))

            # Remove train tuples otherwise maybe leakage?
            relevant_product_edges_no_train = relevant_product_edges.difference(relevant_train_edges_tuples)

            # Write edges to the targets file
            test_edge_name_mapped = map_edge_name(test_edge_name)
            with open(f'{psl_dir}/{test_edge_name_mapped}_targets.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}' for u, v in relevant_product_edges_no_train))

            # Write edges to the truths file
            relevant_test_edges = [e for e in test_edges if e[-1]['edge'] == test_edge_name]
            relevant_product_edges_no_train_negatives = relevant_product_edges_no_train.difference(relevant_test_edges_tuples)
            with open(f'{psl_dir}/{test_edge_name_mapped}_truth.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v in relevant_test_edges_tuples))
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t0' for u, v in relevant_product_edges_no_train_negatives))

        # Just write the test set as the targets and all have truth values of 1
        else:
            with open(f'{psl_dir}/{test_edge_name_mapped}_targets.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}' for u, v in relevant_test_edges_tuples))
            with open(f'{psl_dir}/{test_edge_name_mapped}_truth.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v in relevant_test_edges_tuples))

