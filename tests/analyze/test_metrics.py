"""Tests for calculating metrics about the input and output data for kgemb-sens."""

# -*- coding: utf-8 -*-

import unittest

import networkx as nx
import numpy as np

from kgemb_sens.analyze.metrics import calc_edge_input_statistics, calc_network_input_statistics, calc_powerlaw_statistics, calc_output_statistics
from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts
from kgemb_sens.transform.graph_utilities import undirect_multidigraph

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
        cls.cc3.add_edges_from([(0, 1, {"edge": "blah"})])
        cls.cc3_degree_dict = dict(cls.cc3.degree())

        # Nations
        cls.nations = load_benchmark_data_three_parts("nations", DATA_DIR)

    def test_calc_edge_input_statistics(self):
        e = (0, 1, {"edge": "blah"})
        edge_min_node_degree, edge_rel_count, e_deg = calc_edge_input_statistics(self.cc3, e, self.cc3_degree_dict)
        self.assertEqual(edge_min_node_degree, 7)
        self.assertEqual(edge_rel_count, 0)
        self.assertEqual(e_deg, 10)

        e = (0, 1, {"edge": "test"})
        edge_min_node_degree, edge_rel_count, e_deg = calc_edge_input_statistics(self.cc3, e, self.cc3_degree_dict)
        self.assertEqual(edge_rel_count, 19)

        e = ("s", 0)
        edge_min_node_degree, _, e_deg = calc_edge_input_statistics(self.clg8, e, self.clg8_degree_dict)
        self.assertEqual(edge_min_node_degree, 1)
        self.assertEqual(e_deg, 6)

    def test_calc_network_input_statistics(self):
        n_ent_network, n_rel_network, n_conn_comps, avg_cc, med_rel_count, min_rel_count, diam = calc_network_input_statistics(self.cc3, calc_diam=True)
        self.assertEqual(n_ent_network, 9)
        self.assertEqual(n_rel_network, 2)
        self.assertEqual(n_conn_comps, 3)
        self.assertEqual(diam, float("inf"))
        self.assertAlmostEqual(avg_cc, 0.777777, places=4)
        self.assertEqual(med_rel_count, 10.5)
        self.assertEqual(min_rel_count, 1)

        n_ent_network, n_rel_network, n_conn_comps, avg_cc, med_rel_count, min_rel_count, diam = calc_network_input_statistics(self.clg8, calc_diam=True)
        self.assertEqual(n_ent_network, 18)
        self.assertEqual(n_rel_network, 1)
        self.assertEqual(n_conn_comps, 1)
        self.assertEqual(diam, 6)
        self.assertEqual(avg_cc, 0)
        self.assertEqual(med_rel_count, 50)
        self.assertEqual(min_rel_count, 50)

        n_ent_network, n_rel_network, n_conn_comps, avg_cc, med_rel_count, min_rel_count = calc_network_input_statistics(self.nations)
        self.assertEqual(n_ent_network, 14)
        self.assertEqual(n_rel_network, 55)
        self.assertEqual(n_conn_comps, 1)
        self.assertEqual(avg_cc, 1)
        self.assertEqual(med_rel_count, 23.0)
        self.assertEqual(min_rel_count, 1)

    def test_calc_powerlaw_statistics(self):
        pass  # TODO

    def test_calc_output_statistics(self):
        pass  # TODO


