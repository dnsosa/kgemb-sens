"""Tests for manipulating the input data for kgemb-sens as in the whole pipeline."""

# -*- coding: utf-8 -*-

import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts
from kgemb_sens.transform.graph_utilities import edge_dist, undirect_multidigraph
from kgemb_sens.transform.processing_pipeline import graph_processing_pipeline

# from .resources.test_processing_pipeline_helpers import num_

DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"


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

    @unittest.skip("Takes a while to process Nations, don't need to test all the time")
    def test_graph_processing_pipeline_sparsify(self):

        out_dir = os.path.join(os.path.dirname(__file__), "test_out_dir")
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

