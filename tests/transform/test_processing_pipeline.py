"""Tests for manipulating the input data for kgemb-sens as in the whole pipeline."""

# -*- coding: utf-8 -*-

import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from collections import Counter

from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts, load_drkg_data
from kgemb_sens.transform.graph_utilities import edge_dist, undirect_multidigraph, edge_degree
from kgemb_sens.transform.contradiction_utilities import find_all_valid_negations
from kgemb_sens.transform.processing_pipeline import graph_processing_pipeline
from kgemb_sens.utilities import good_round

DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"
out_dir = os.path.join(os.path.dirname(__file__), "test_out_dir")


class TestProcessingPipeline(unittest.TestCase):
    """Tests for graph utilities in sparsification pipeline for kgemb-sens."""

    @classmethod
    def setUpClass(cls):
        # Initialize network structure
        # Circular ladder graph with two leaves
        cls.clg8 = nx.MultiDiGraph(nx.circular_ladder_graph(8))
        cls.clg8.add_edges_from([("s", 0), (4, "t")])
        cls.clg8_dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(undirect_multidigraph(cls.clg8))))
        cls.clg8_degree_dict = dict(cls.clg8.degree())
        attrs_dict = {}
        for e in cls.clg8.edges(keys=True):
            attrs_dict[e] = {"edge": "test"}
        nx.set_edge_attributes(cls.clg8, attrs_dict)

        # Graph of 3 complete connected components: size 4, 3, and 2. Attributes added including one double attribute
        cls.cc3 = nx.compose_all([nx.complete_graph(range(0, 4)),
                                  nx.complete_graph(range(4, 7)),
                                  nx.complete_graph(range(7, 9))])
        cls.cc3 = nx.MultiDiGraph(cls.cc3)
        cls.cc3_dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(undirect_multidigraph(cls.cc3))))
        attrs_dict = {}
        for e in cls.cc3.edges(keys=True):
            attrs_dict[e] = {"edge": "test"}
        nx.set_edge_attributes(cls.cc3, attrs_dict)
        cls.cc3.add_edges_from([(0, 1, {"edge": "blah"})])
        cls.cc3_degree_dict = dict(cls.cc3.degree())

        # Nations
        cls.nations = load_benchmark_data_three_parts("nations", DATA_DIR)
        cls.nations_degree_dict = dict(cls.nations.degree())

    @unittest.skip("Takes a while to process Nations, don't need to test all the time")
    def test_graph_processing_pipeline_sparsify(self):

        # alpha 0, -1, 1
        # prob_type = degree, distance

        # Test number of edges
        # test at least one of edge type in train
        # test at least one of each node in train
        # Test something about how distance and degree work

        # then test nations
        # Test number of rows in the output directories

        # Distance ones -- clg8
        params = {"dataset": None,
                  "pcnet_filter": None,
                  "val_test_frac": 1,  # Not changing this for now. TODO: test how this affects things
                  "val_frac": 0,
                  "sparsified_frac": 0.25,
                  "alpha": 0,
                  "n_resample": 1,
                  "prob_type": "distance",  # distance, degree
                  "flatten_kg": False,  # Not changing this for now. TODO: test how this affects things
                  "neg_completion_frac": 0.5,  # Wanna show this has no effect
                  "contradiction_frac": 0.5,  # Wanna show this has no effect
                  "contra_remove_frac": 0.5,  # Wanna show this has no effect
                  "MODE": "sparsification",  # "sparsification", "contradictification", "contrasparsify"
                  "model_name": None,
                  "n_epochs": None}

        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.cc3, 0, params, out_dir)
        test_edge = edge_divisions[1][0]
        # Check test edge not in train or sparse edges
        self.assertTrue(test_edge not in edge_divisions[0])
        self.assertTrue(test_edge not in edge_divisions[2])
        # Check test edge in the same connected component as sparse edges
        for edge in edge_divisions[2]:
            self.assertTrue(nx.has_path(self.cc3, test_edge[0], edge[0]))

        params["alpha"] = -2
        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.clg8, 0, params, out_dir)
        test_edge = edge_divisions[1][0]
        sparse_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in edge_divisions[2]]
        train_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in edge_divisions[0]]
        self.assertTrue(np.mean(sparse_dists) < np.mean(train_dists))

        params["alpha"] = 100
        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.clg8, 0, params, out_dir)
        test_edge = edge_divisions[1][0]
        sparse_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in edge_divisions[2]]
        self.assertGreaterEqual(np.mean(sparse_dists), 4)

        # Degree ones -- cc3
        params["prob_type"] = "degree"
        params["sparsified_frac"] = 0.25
        params["alpha"] = -4
        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.cc3, 0, params, out_dir)
        test_edge = edge_divisions[1][0]
        self.assertTrue(test_edge not in edge_divisions[0])
        self.assertTrue(test_edge not in edge_divisions[2])
        # Want to show that the sparse edges land in multiple components
        sparse_nodes = set([u for u, _, _, _ in edge_divisions[2]] + [v for _, v, _, _ in edge_divisions[2]])
        self.assertTrue(8 in sparse_nodes)
        self.assertNotEqual(set([0, 1, 2, 3]).intersection(sparse_nodes), 0)
        self.assertNotEqual(set([4, 5, 6]).intersection(sparse_nodes), 0)

        params["alpha"] = 100
        params["sparsified_frac"] = 0.05
        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.cc3, 0, params, out_dir)
        sparse_nodes = set([u for u, _, _, _ in edge_divisions[2]] + [v for _, v, _, _ in edge_divisions[2]])
        self.assertNotEqual(set([0, 1, 2, 3]).intersection(sparse_nodes), 0)

        # Nations
        params["alpha"] = 1
        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.nations, 0, params, out_dir)
        test_edge = edge_divisions[1][0]
        self.assertTrue(test_edge[0] in G_out.nodes())
        self.assertTrue(test_edge[1] in G_out.nodes())
        self.assertTrue(test_edge[3]['edge'] in [rel for _, _, rel in G_out.edges(data="edge")])
        # Check number of edges removed
        self.assertEqual(G_out.number_of_edges(), round(self.nations.number_of_edges()*0.95))

        params["sparsified_frac"] = 0.8
        data_paths, train_conditions_id, edge_divisions, G_out = graph_processing_pipeline(self.nations, 0, params, out_dir)
        test_edge = edge_divisions[1][0]
        self.assertTrue(test_edge[0] in G_out.nodes())
        self.assertTrue(test_edge[1] in G_out.nodes())
        self.assertTrue(test_edge[3]['edge'] in [rel for _, _, rel in G_out.edges(data="edge")])
        # Check number of edges removed
        self.assertEqual(G_out.number_of_edges(), round(self.nations.number_of_edges()*0.2))

        # Read the files, check that the right number of edges are there
        output_train_path = f"{out_dir}/train_{train_conditions_id}.tsv"
        output_test_path = f"{out_dir}/test_{train_conditions_id}.tsv"
        self.assertEqual(len(pd.read_csv(output_train_path, sep='\t', header=None)), round(self.nations.number_of_edges()*0.2)-1)
        self.assertEqual(len(pd.read_csv(output_test_path, sep='\t', header=None)), 2)  # 2 edges is the little hack

    @unittest.skip("Takes a while to process Nations, don't need to test all the time")
    def test_graph_processing_pipeline_contradictification(self):
        # Distance ones -- clg8
        params = {"dataset": None,
                  "pcnet_filter": None,
                  "val_test_frac": 1,  # Not changing this for now. TODO: test how this affects things
                  "val_frac": 0,
                  "sparsified_frac": 0.25,  # Wanna show this has no effect
                  "alpha": 10,
                  "n_resample": 1,
                  "prob_type": "distance",  # distance, degree
                  "flatten_kg": False,  # Not changing this for now. TODO: test how this affects things
                  "neg_completion_frac": 0,
                  "contradiction_frac": 0.5,
                  "contra_remove_frac": 0,
                  "MODE": "contrasparsify",
                  "model_name": None,
                  "n_epochs": None}

        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.clg8, 0, params, out_dir,
                                                                         edge_names=["test"],
                                                                         dist_mat=self.clg8_dist_mat)
        test_edge = edge_divisions[1][0]
        new_contradictions, removed_contradictions = edge_divisions[3], edge_divisions[4]
        contra_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in new_contradictions]
        train_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in edge_divisions[0]]

        contra_edges_nodes = set([u for u, _, _, _ in new_contradictions] + [v for _, v, _, _ in new_contradictions])
        G_out_rel_counter = Counter([r for _, _, _, r in G_out.edges(data='edge', keys=True)])
        G_out_key_counter = Counter([k for _, _, k, _ in G_out.edges(data='edge', keys=True)])

        self.assertEqual(G_out.number_of_edges(), 75)
        self.assertEqual(G_out_rel_counter["NOT-test"], 25)
        self.assertEqual(G_out_key_counter[1], 25)

        self.assertTrue(np.mean(contra_dists) > np.mean(train_dists))
        self.assertTrue(4 in contra_edges_nodes)  # with high probability
        self.assertEqual(len(removed_contradictions), 0)
        self.assertTrue(edge_divisions[2] is None)

        # Test contradiction of close edges
        params["alpha"] = -2
        params["contra_remove_frac"] = 1  # See if it does the resample

        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.clg8, 0, params, out_dir,
                                                                         edge_names=["test"],
                                                                         dist_mat=self.clg8_dist_mat)
        test_edge = edge_divisions[1][0]
        new_contradictions, removed_contradictions = edge_divisions[3], edge_divisions[4]
        contra_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in new_contradictions]
        train_dists = [edge_dist(test_edge, sparse_edge, self.clg8_dist_mat) for sparse_edge in edge_divisions[0]]

        G_out_rel_counter = Counter([r for _, _, _, r in G_out.edges(data='edge', keys=True)])
        G_out_key_counter = Counter([k for _, _, k, _ in G_out.edges(data='edge', keys=True)])

        self.assertEqual(G_out.number_of_edges(), 25)
        self.assertFalse("NOT-test" in G_out_key_counter)
        self.assertFalse(1 in G_out_rel_counter)

        self.assertTrue(np.mean(contra_dists) < np.mean(train_dists))
        self.assertEqual(len(new_contradictions), 50)
        self.assertEqual(len(removed_contradictions), 50)
        self.assertTrue(edge_divisions[2] is None)

        # Degree one -- cc3
        # need to calculate AVN
        params["prob_type"] = "degree"
        params["neg_completion_frac"] = 0.1
        params["contradiction_frac"] = 1.0
        params["contra_remove_frac"] = 0.25

        avn_cc3 = find_all_valid_negations(self.cc3)
        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.cc3, 0, params, out_dir,
                                                                         all_valid_negations=avn_cc3,
                                                                         edge_names=["test", "blah"],
                                                                         degree_dict=self.cc3_degree_dict)
        test_edge = edge_divisions[1][0]
        new_contradictions, removed_contradictions = edge_divisions[3], edge_divisions[4]
        contra_edges_nodes = set([u for u, _, _, _ in new_contradictions] + [v for _, v, _, _ in new_contradictions])
        G_out_rel_counter = Counter([r for _, _, _, r in G_out.edges(data='edge', keys=True)])

        self.assertEqual(G_out.number_of_edges(), 35)  # 20 * 1.1 = 22 - 1 test * .25 = remove 5; 1 * 1.1 * .25 = remove 0
        self.assertEqual(G_out_rel_counter["NOT-test"], 16)  # 21 - 5
        self.assertTrue(4 in contra_edges_nodes)  # with high probability
        self.assertEqual(len(removed_contradictions), 10)  # removed 5
        self.assertTrue(edge_divisions[2] is None)

        # One Nations test
        params["prob_type"] = "degree"
        params["alpha"] = 0
        params["neg_completion_frac"] = 0
        params["contradiction_frac"] = 0.1
        params["contra_remove_frac"] = 0.5

        edge_names = ["exports3", "embassy", "accusation"]  # 34, 141, 23
        # Starting edges: 1992
        # Contradictions: +3, +14, +2
        # Remove fraction: -4, -14, -2
        # End: 1991

        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(self.nations, 0, params, out_dir,
                                                                         edge_names=edge_names,
                                                                         degree_dict=self.nations_degree_dict)

        G_out_rel_counter = Counter([r for _, _, _, r in G_out.edges(data='edge', keys=True)])
        G_out_rels = set([r for _, _, _, r in G_out.edges(data='edge', keys=True)])
        nations_rels = set([r for _, _, _, r in self.nations.edges(data='edge', keys=True)])
        self.assertEqual(G_out.number_of_edges(), 1991)
        self.assertEqual(len(G_out_rels), len(nations_rels) + 3)
        self.assertEqual(G_out_rel_counter["exports3"], 32)
        self.assertEqual(G_out_rel_counter["NOT-exports3"], 1)
        self.assertEqual(G_out_rel_counter["embassy"], 134)
        self.assertEqual(G_out_rel_counter["NOT-embassy"], 7)
        self.assertEqual(G_out_rel_counter["accusation"], 22)
        self.assertEqual(G_out_rel_counter["NOT-accusation"], 1)

    def test_graph_processing_pipeline_contradictification_gnbr(self):
        # One GNBR test
        params = {"dataset": "gnbr_drg",
                  "pcnet_filter": None,
                  "val_test_frac": 1,
                  "val_frac": 0,
                  "sparsified_frac": 0.25,
                  "alpha": 0,
                  "n_resample": 1,
                  "prob_type": "degree",
                  "flatten_kg": False,
                  "neg_completion_frac": 0,
                  "contradiction_frac": 0.5,
                  "contra_remove_frac": 0,
                  "MODE": "contrasparsify",
                  "model_name": None,
                  "n_epochs": None,
                  "vt_alpha": 10}

        G_gnbr_drg = load_drkg_data("gnbr_drg", DATA_DIR)
        G_gnbr_drg_undir = undirect_multidigraph(G_gnbr_drg)
        edge_names = set([r for _, _, r in G_gnbr_drg.edges(data='edge')])
        antonyms = [("E+", "E-"), ("A+", "A-")]
        G_gnbr_drg_degree_dict = dict(G_gnbr_drg.degree())

        data_paths, _, edge_divisions, G_out = graph_processing_pipeline(G_gnbr_drg, 0, params, out_dir,
                                                                         edge_names=edge_names, antonyms=antonyms,
                                                                         degree_dict=G_gnbr_drg_degree_dict)

        test_edge = edge_divisions[1][0]
        G_out_rel_counter = Counter([r for _, _, _, r in G_out.edges(data='edge', keys=True)])
        G_in_rel_counter = Counter([r for _, _, _, r in G_gnbr_drg.edges(data='edge', keys=True)])
        e_deg = edge_degree(G_gnbr_drg_undir, test_edge, G_gnbr_drg_degree_dict)
        e_degs = [edge_degree(G_gnbr_drg_undir, e, G_gnbr_drg_degree_dict) for e in G_gnbr_drg.edges(data='edge')]

        self.assertEqual(G_out_rel_counter["E+"], G_in_rel_counter["E+"] + good_round(.5*G_in_rel_counter["E-"]))
        self.assertEqual(G_out_rel_counter["A-"], G_in_rel_counter["A-"] + good_round(.5*G_in_rel_counter["A+"]))
        self.assertEqual(G_out_rel_counter["N"], G_in_rel_counter["N"])

        rank_e_deg = sorted(e_degs, reverse=True).index(e_deg)
        # 80,000 triples, this should sample a high-degree triple. Problem before with vt_alpha=1000 was related to
        # issues of numerical precision/overflow
        self.assertTrue(rank_e_deg < 1000)
