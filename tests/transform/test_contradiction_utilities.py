"""Tests the helper functions for the contradictions pipeline."""

# -*- coding: utf-8 -*-

import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts
from kgemb_sens.transform.graph_utilities import edge_dist, undirect_multidigraph
from kgemb_sens.transform.processing_pipeline import graph_processing_pipeline
from kgemb_sens.transform.contradiction_utilities import find_all_valid_negations, negative_completion,\
    generate_converse_edges_from, fill_with_contradictions, remove_contradictions

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

    def test_negative_completion(self):
        rels_cc3 = set([r for _, _, r in self.cc3.edges(data='edge')])
        n_rel_cc3 = len(rels_cc3)
        n_rel_clg8 = len(set([r for _, _, r in self.clg8.edges(data='edge')]))

        # Find all valid negations
        avn_cc3 = find_all_valid_negations(self.cc3)
        avn_clg8 = find_all_valid_negations(self.clg8)
        # TODO: What happens in the case where there's no edge attribute?

        # Test for the right number of edges
        self.assertEqual(len(avn_cc3), 123)  # (9C2 * 2 - 20) + (9C2 * 2 - 1)
        self.assertEqual(len(avn_clg8), 256)  # (18C2 * 2 - (24 * 2 + 2))
        rels_avn_cc3 = set([r for _, _, r in avn_cc3])
        n_rel_avn_cc3 = len(rels_avn_cc3)
        n_rel_avn_clg8 = len(set([r for _, _, r in avn_clg8]))
        self.assertEqual(n_rel_avn_cc3, n_rel_cc3)
        self.assertEqual(n_rel_avn_clg8, n_rel_clg8)

        # Test the right edges are added
        for rel in rels_cc3:
            self.assertTrue(f"NOT-{rel}" in rels_avn_cc3)

        # Negative completion fraction
        cc3_nc = negative_completion(self.cc3, avn_cc3, 1.0)
        clg8_nc = negative_completion(self.clg8, avn_cc3, 0.2)
        rels_cc3_nc = set([r for _, _, r in cc3_nc.edges(data='edge')])

        # Test that the right predicates are added
        for rel in rels_cc3:
            self.assertTrue(rel in rels_cc3_nc)
            self.assertTrue(f"NOT-{rel}" in rels_cc3_nc)

        # Test expected number of edges
        self.assertEqual(cc3_nc.number_of_edges(), self.cc3.number_of_edges()*2)
        self.assertEqual(clg8_nc.number_of_edges(), round(self.clg8.number_of_edges()*1.2))

    def test_generate_converse_edges_from(self):
        pass

    def test_fill_with_contradictions(self):
        pass

    def test_remove_contradictions(self):
        pass