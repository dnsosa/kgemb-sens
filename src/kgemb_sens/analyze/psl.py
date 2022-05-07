# -*- coding: utf-8 -*-

"""Run the embedding pipeline."""

import itertools
import os
import random

import numpy as np
import pandas as pd

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from kgemb_sens.analyze.metrics import calc_edge_input_statistics, calc_network_input_statistics, calc_output_statistics
from kgemb_sens.transform.graph_utilities import random_split_list, undirect_multidigraph


def run_psl_pipeline(data_paths, i, params, train_conditions_id, G, test_edges, train_edges, out_dir,
                       degree_dict=None, G_undir=None, antonyms=None):

    psl_dir = f"{out_dir}/{params['dataset']}_psl"

    create_data_files_for_psl(G, train_edges=train_edges, test_edges=test_edges, psl_dir=psl_dir,
                              num_negatives=int(params["n_negatives"]), full_product=params["full_product"],
                              dataset=params["dataset"])

    ADDITIONAL_PSL_OPTIONS = {
        'log4j.threshold': 'INFO',
        #'infer': 'org.linqs.psl.application.inference.LazyMPEInference',
        #'eval': 'org.linqs.psl.evaluation.statistics.DiscreteEvaluator'
    }

    model = Model(params["dataset"])

    # Add predicates
    train_edge_names = set([r['edge'] for _, _, _, r in train_edges])
    test_edge_names = set([r['edge'] for _, _, _, r in test_edges])
    predicates_list = add_predicates(model, train_edge_names, test_edge_names)

    # Add rules
    create_psl_rules(model, params["dataset"], psl_dir, antonyms, psl_contras=params["psl_contras"])
    print("Rules are done!")

    # Weight learning
    #learn_weights(model, predicates_list, psl_dir,
    #              psl_options=ADDITIONAL_PSL_OPTIONS)

    for rule in model.get_rules():
        print('\n' + str(rule))

    # Inference
    results = psl_infer(model, predicates_list, psl_dir,
                        psl_options=ADDITIONAL_PSL_OPTIONS)

    write_results(results, model, psl_dir)


def map_edge_name(in_edge_name):
    edge_name_map = {"A+": "Ap",
                     "A-": "An",
                     "E+": "Ep",
                     "E-": "En",
                     "V+": "Vp",
                     "Gr>G": "GrrG"}

    if in_edge_name in edge_name_map:
        return edge_name_map[in_edge_name]
    else:
        return in_edge_name


def create_data_files_for_psl(G, train_edges, test_edges, psl_dir, num_negatives, full_product, dataset):

    # open the file, write the first line about the observed. Make note of which predicates will be open.
    # Prepare mappings and general compatibilities with PSL expectations
    os.makedirs(psl_dir, exist_ok=True)
    psl_learn_dir = f"{psl_dir}/learn"
    os.makedirs(psl_learn_dir, exist_ok=True)
    psl_eval_dir = f"{psl_dir}/eval"
    os.makedirs(psl_eval_dir, exist_ok=True)

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
        all_nodes = list(set(list(itertools.chain(*relevant_edges_node_tuples))))
        # head_nodes = set([u for u, _ in relevant_edges_node_tuples])
        # tail_nodes = set([v for _, v in relevant_edges_node_tuples])
        # NOTE: Should think about if when I corrupt head, I only resample other head nodes (i.e. just drugs),
        # or any node, i.e. could be a disease

        # Repeat procedure until have enough negative samples
        negative_edges = set([])
        while len(negative_edges) < (len(relevant_edges) * num_negatives):
            sampled_edge = random.choice(relevant_edges)
            if np.random.binomial(1, .5):  # flip a coin
                # head corrupt
                corrupt_edge = (random.choice(all_nodes), sampled_edge[1])
            else:
                # tail corrupt
                corrupt_edge = (sampled_edge[0], random.choice(all_nodes))

            if (corrupt_edge not in relevant_edges_node_tuples) and (corrupt_edge not in negative_edges):
                negative_edges.add(corrupt_edge)

        # Write edges
        # Note, for now writing to both learn and eval directories
        edge_name_mapped = map_edge_name(edge_name)
        with open(f'{psl_learn_dir}/{edge_name_mapped}_obs.txt', 'w') as fp:
            # Positive examples
            fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v, _, _ in relevant_edges))
            fp.write("\n")
            # Negative examples
            fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t0' for u, v in negative_edges))
        with open(f'{psl_eval_dir}/{edge_name_mapped}_obs.txt', 'w') as fp:
            # Positive examples
            fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v, _, _ in relevant_edges))
            fp.write("\n")
            # Negative examples
            fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t0' for u, v in negative_edges))

    # Now make the target and truth files...
    # Target file
    test_edge_names = set([r['edge'] for _, _, _, r in test_edges])
    for test_edge_name in test_edge_names:

        test_edge_name_mapped = map_edge_name(test_edge_name)
        relevant_test_edges = [e for e in test_edges if e[-1]['edge'] == test_edge_name]
        relevant_test_edges_tuples = set([(u, v) for u, v, _, _ in relevant_test_edges])
        relevant_test_edges_tuples_val, relevant_test_edges_tuples_test = random_split_list(list(relevant_test_edges_tuples), 0.5)

        # Take the product of the head nodes for the edge type by the tail nodes for the edge type
        if full_product:
            relevant_train_edges = [e for e in train_edges if e[-1]['edge'] == test_edge_name]
            relevant_train_edges_tuples = set([(u, v) for u, v, _, _ in relevant_train_edges])
            train_edge_heads = [u for u, _ in relevant_train_edges]
            train_edge_tails = [v for _, v in relevant_train_edges]
            relevant_product_edges = set(list(itertools.product(train_edge_heads, train_edge_tails)))

            # Remove train tuples otherwise maybe leakage?
            relevant_product_edges_no_train = relevant_product_edges.difference(relevant_train_edges_tuples)
            relevant_product_edges_no_train_val, relevant_product_edges_no_train_test = random_split_list(list(relevant_product_edges_no_train), 0.5)

            # Write edges to the targets file
            with open(f'{psl_learn_dir}/{test_edge_name_mapped}_targets.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}' for u, v in relevant_product_edges_no_train_val))
            with open(f'{psl_eval_dir}/{test_edge_name_mapped}_targets.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}' for u, v in relevant_product_edges_no_train_test))

            # Write edges to the truths file
            relevant_product_edges_no_train_negatives = relevant_product_edges_no_train.difference(relevant_test_edges_tuples)
            relevant_product_edges_no_train_negatives_val, relevant_product_edges_no_train_negatives_test = random_split_list(list(relevant_product_edges_no_train_negatives), 0.5)
            with open(f'{psl_learn_dir}/{test_edge_name_mapped}_truth.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v in relevant_test_edges_tuples_val))
                fp.write("\n")
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t0' for u, v in relevant_product_edges_no_train_negatives_val))
            with open(f'{psl_eval_dir}/{test_edge_name_mapped}_truth.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v in relevant_test_edges_tuples_test))
                fp.write("\n")
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t0' for u, v in relevant_product_edges_no_train_negatives_test))

        # Just write the test set as the targets and all have truth values of 1
        else:
            with open(f'{psl_learn_dir}/{test_edge_name_mapped}_targets.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}' for u, v in relevant_test_edges_tuples_val))
            with open(f'{psl_eval_dir}/{test_edge_name_mapped}_targets.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}' for u, v in relevant_test_edges_tuples_test))

            with open(f'{psl_learn_dir}/{test_edge_name_mapped}_truth.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v in relevant_test_edges_tuples_val))
            with open(f'{psl_eval_dir}/{test_edge_name_mapped}_truth.txt', 'w') as fp:
                fp.write('\n'.join(f'{node_map[u]}\t{node_map[v]}\t1' for u, v in relevant_test_edges_tuples_test))

    # Write the data file
    closed_edge_names = edge_names.difference(test_edge_names)
    for dir in [psl_learn_dir, psl_eval_dir]:
        with open(f'{dir}/{dataset}.data', 'w') as fp:
            fp.write("predications:\n")
            fp.write('\n'.join(f'   {map_edge_name(edge_name)}/2: closed' for edge_name in closed_edge_names))
            fp.write("\n")
            fp.write('\n'.join(f'   {map_edge_name(edge_name)}/2: open' for edge_name in test_edge_names))
            fp.write('\n\n')
            fp.write("observations:\n")
            fp.write('\n'.join(f'   {dir}/{map_edge_name(edge_name)}_obs.txt' for edge_name in edge_names))
            fp.write('\n\n')
            fp.write("targets:\n")
            fp.write('\n'.join(f'   {dir}/{map_edge_name(edge_name)}_targets.txt' for edge_name in test_edge_names))
            fp.write('\n\n')
            fp.write("truth:\n")
            fp.write('\n'.join(f'   {dir}/{map_edge_name(edge_name)}_truth.txt' for edge_name in test_edge_names))


def add_predicates(model, train_edge_names, test_edge_names):

    predicates_list = []
    closed_edge_names = train_edge_names.difference(test_edge_names)
    for closed_edge_name in closed_edge_names:
        closed_edge_name_mapped = map_edge_name(closed_edge_name)
        new_closed_predicate = Predicate(closed_edge_name_mapped, closed=True, size=2)
        model.add_predicate(new_closed_predicate)
        predicates_list.append(new_closed_predicate)

    for open_edge_name in test_edge_names:
        open_edge_name_mapped = map_edge_name(open_edge_name)
        new_open_predicate = Predicate(open_edge_name_mapped, closed=False, size=2)
        model.add_predicate(new_open_predicate)
        predicates_list.append(new_open_predicate)

    return predicates_list


def create_psl_rules(model, dataset, psl_dir, antonyms, psl_contras):
    # gnbr
    if dataset == "gnbr":
        drg_rels = ["Ap", "An", "B", "Ep", "En", "E", "N", "O", "K", "Z"]
        gg_rels = ["B", "W", "Vp", "Ep", "E", "I", "H", "Rg", "Q"]
        gdz_rels = ["Md", "X", "L", "U", "Ud", "D", "J", "Te", "Y", "G"]
        drdz_rels = ["T", "C", "Sa", "Pr", "Pa", "J", "Mp"]
        treats_rels = ["T", "Pa"]

    # hetionet
    else:
        drg_rels = ["CuG", "CdG", "CbG"]
        gg_rels = ["GcG", "GiG", "GrrG"]
        gdz_rels = ["DaG", "DdG", "DuG"]
        drdz_rels = ["CtD", "CpD"]
        treats_rels = ["CtD", "CpD"]

    # DrGDz Paths
    drg_gdz_tuples = set(list(itertools.product(drg_rels, gdz_rels, treats_rels)))
    drg_gg_gdz_tuples = set(list(itertools.product(drg_rels, gg_rels, gdz_rels, treats_rels)))
    drdz_dzdr_drdz_tuples = set(
        list(itertools.product(drdz_rels, drdz_rels, drdz_rels, treats_rels)))  # Note weirdness with the directionality
    triple_non_equality = "(A != B) & (B != C) & (A != C)"
    quad_non_equality = triple_non_equality + "& (A != D) & (B != D) & (C != D)"
    drgdz_rules = [f"10: {r1}(A, B) & {r2}(B, C) & {triple_non_equality} -> {t}(A, C) ^2" for (r1, r2, t) in
                   drg_gdz_tuples]
    drggdz_rules = [f"10: {r1}(A, B) & {r2}(B, C) & {r3}(C, D) & {quad_non_equality} -> {t}(A, D) ^2" for
                    (r1, r2, r3, t) in drg_gg_gdz_tuples]
    drdzdrdz_rules = [f"10: {r1}(A, B) & {r2}(B, C) & {r3}(C, D) & {quad_non_equality} -> {t}(A, D) ^2" for
                      (r1, r2, r3, t) in drdz_dzdr_drdz_tuples]

    ##all_rules = drgdz_rules + drggdz_rules + drdzdrdz_rules
    all_rules = drgdz_rules
    if psl_contras:
        pair_non_equality = "(A != B)"
        contra_rules = [f"10: {map_edge_name(r1)}(A, B) & {pair_non_equality} -> !{map_edge_name(r2)}(A, B) ^2" for (r1, r2) in antonyms]
        contra_rules += [f"10: {map_edge_name(r2)}(A, B) & {pair_non_equality} -> !{map_edge_name(r1)}(A, B) ^2" for (r1, r2) in antonyms]
        all_rules += contra_rules

    # Add the priors rules -- start from the prior that a drug doesn't treat a disease
    neg_prior = [f"1: !{t}(X, Y) ^2" for t in treats_rels]
    all_rules += neg_prior

    for rule in all_rules:
        model.add_rule(Rule(rule))

    # Write rules to a .psl file
    with open(f"{psl_dir}/{dataset}.psl", 'w') as fp:
        fp.write('\n'.join(f'{rule}' for rule in all_rules))


def add_data(split, predicates_list, psl_dir):
    split_data_dir = f"{psl_dir}/{split}"

    for predicate in predicates_list:
        predicate.clear_data()

        predicate_name = predicate.name()
        obs_path = f'{split_data_dir}/{predicate_name}_obs.txt'
        predicate.add_data_file(Partition.OBSERVATIONS, obs_path)

        if not predicate.closed():
            targets_path = f'{split_data_dir}/{predicate_name}_targets.txt'
            print(f"Adding target data from {targets_path} ...")
            predicate.add_data_file(Partition.TARGETS, targets_path)

            truth_path = f'{split_data_dir}/{predicate_name}_truth.txt'
            print(f"Adding truth data from {truth_path} ...")
            predicate.add_data_file(Partition.TRUTH, truth_path)
            print(f"Done with predicate =={predicate_name}==")


def learn_weights(model, predicates_list, psl_dir, psl_options):
    add_data('learn', predicates_list, psl_dir)
    #model.learn(additional_cli_optons=cli_options, psl_config=psl_options)  # optons [sic]
    model.learn(method='org.linqs.psl.application.inference.LazyMPEInference',
                psl_config=psl_options)  # optons [sic]


def psl_infer(model, predicates_list, psl_dir, psl_options):
    add_data('eval', predicates_list, psl_dir)
    #return model.infer(additional_cli_optons=cli_options, psl_config=psl_options)  # optons [sic]
    return model.infer(method='org.linqs.psl.application.inference.LazyMPEInference',
                       psl_config=psl_options)  # optons [sic]


def write_results(results, model, psl_dir):
    out_dir = f'{psl_dir}/inferred_predicates'
    os.makedirs(out_dir, exist_ok=True)

    for predicate in model.get_predicates().values():
        if predicate.closed():
            continue

        out_path = f"{out_dir}/{predicate.name()}.txt"
        results[predicate].to_csv(out_path, sep="\t", header=False, index=False)
