"""Tests for manipulating the input data for kgemb-sens."""

# -*- coding: utf-8 -*-

import unittest

import networkx as nx

from kgemb_sens.transform.graph_utilities import edge_dist, edge_degree, prob_dist, random_split_list, undirect_multidigraph
from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts

DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"


class TestGraphUtilities(unittest.TestCase):
    """Tests for graph utilities in sparsification pipeline for kgemb-sens."""

    @classmethod
    def setUpClass(cls):
        # Initialize network structure
        # Circular ladder graph with two leaves
        cls.clg8 = nx.MultiDiGraph(nx.circular_ladder_graph(8))
        cls.clg8.add_edges_from([("s", 0), (4, "t")])
        cls.clg8_dist_mat = dict(nx.all_pairs_bellman_ford_path_length(nx.Graph(undirect_multidigraph(cls.clg8))))
        cls.clg8_degree_dict = dict(cls.clg8.degree())

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
        cls.cc3.add_edges_from([(0, 1, {'edge': 'blah'})])
        cls.cc3_degree_dict = dict(cls.cc3.degree())
        print(cls.cc3.edges(keys=True, data=True))

        ### Nations
        ##cls.nations = load_benchmark_data_three_parts("nations", DATA_DIR)
        ##cls.assertEqual(cls.nations.number_of_edges(), 1992)

    def test_edge_dist(self):
        # Circular ladder
        # Optional To-Do: test edges not in graph. Now that it's directed... could become dicey. Making assumptions
        # about when it is and isn't directed sometimes.
        dist1 = edge_dist(("s", 0), (4, "t"), self.clg8_dist_mat)
        self.assertEqual(dist1, 5)
        dist2 = edge_dist(("s", 0), (0, 1), self.clg8_dist_mat)
        self.assertEqual(dist2, 1)
        dist3 = edge_dist(("s", 0), ("s", 0), self.clg8_dist_mat)
        self.assertEqual(dist3, 0)
        dist4 = edge_dist((0, "s"), ("s", 0), self.clg8_dist_mat)
        self.assertEqual(dist4, 0)
        dist5 = edge_dist((0, "s", {"edge": "Test"}), ("s", 0, {"edge": "NOT-Test"}), self.clg8_dist_mat)
        self.assertEqual(dist5, 1)
        dist6 = edge_dist((0, "s", {"edge": "Test"}), ("s", 0), self.clg8_dist_mat)
        self.assertEqual(dist6, 0)

        # Disconnected components graph
        dist7 = edge_dist((0, 1, {"edge": "test"}), (7, 8, {"edge": "test"}), self.cc3_dist_mat)
        self.assertEqual(dist7, float("inf"))

    def test_edge_degree(self):
        deg1 = edge_degree(self.clg8, (0, "s", {"edge": "Test"}), self.clg8_degree_dict)
        self.assertEqual(deg1, 6)
        deg2 = edge_degree(self.cc3, (0, 1, {"edge": "blah"}), self.cc3_degree_dict)
        self.assertEqual(deg2, 10)
        # TODO: Note: not in graph. Behavior?
        deg3 = edge_degree(self.cc3, (2, 3, {"edge": "HECK"}), self.cc3_degree_dict)
        self.assertEqual(deg3, 9)
        deg4 = edge_degree(self.cc3, (8, 7, {"edge": "AnotherHeck"}), self.cc3_degree_dict)
        self.assertEqual(deg4, 1)

    def test_prob_dist(self):
        pass

    def test_random_split_list(self):
        example_list = list(range(10))
        split1L, split1R = random_split_list(example_list, 0.5)
        split2L, split2R = random_split_list(example_list, 0.7)
        split3L, split3R = random_split_list(example_list, 0.3)
        split4L, split4R = random_split_list(example_list, 0)

        self.assertEqual(len(split1L), 5)
        self.assertEqual(len(split1R), 5)
        self.assertEqual(len(split2L), 7)
        self.assertEqual(len(split2R), 3)
        self.assertEqual(len(split3L), 3)
        self.assertEqual(len(split3R), 7)
        self.assertEqual(len(split4L), 0)
        self.assertEqual(len(split4R), 10)
        self.assertTrue(split4L != example_list)  # True with very high probability
