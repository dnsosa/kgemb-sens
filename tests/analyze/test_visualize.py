"""Tests for calculating metrics about the input and output data for kgemb-sens."""

# -*- coding: utf-8 -*-

import unittest
import os.path

import networkx as nx

from kgemb_sens.analyze.visualize import plot_graph_nice

SAVE_DIR = "/Users/dnsosa/Desktop/AltmanLab/KGEmbSensitivity/kgemb-sens/tests/analyze/analyze_test_out"


class TestVisualize(unittest.TestCase):
    """Tests for making a nice network plot for kgemb-sens."""

    def test_plot_graph_nice(self):

        GTest = nx.MultiDiGraph()
        GTest.add_edges_from([('a', 'b', {"edge": "in"}),
                              ('b', 'c', {"edge": "in"}),
                              ('c', 'd', {"edge": "in"}),
                              ('d', 'e', {"edge": "in"}),
                              ('e', 'f', {"edge": "in"}),
                              ('f', 'g', {"edge": "in"}),
                              ('g', 'h', {"edge": "in"}),
                              ('h', 'i', {"edge": "in"}),
                              ('i', 'j', {"edge": "in"}),
                              ('j', 'a', {"edge": "in"}),
                              ('a', 'c', {"edge": "in"}),
                              ('a', 'g', {"edge": "in"}),
                              # ('b', 'c', {"edge": "under"}),
                              # ('b', 'c', {"edge": "over"}),
                              # ('b', 'c', {"edge": "around"}),
                              # ('b', 'c', {"edge": "through"}),
                              # ('b', 'c', {"edge": "within"}),
                              # ('b', 'c', {"edge": "beside"}),
                              ('g', 'j', {"edge": "in"})]
                             )

        train_subset = [('b', 'c', {"edge": "in"}),
                        ('c', 'd', {"edge": "in"}),
                        ('d', 'e', {"edge": "in"}),
                        ('e', 'f', {"edge": "in"}),
                        ('f', 'g', {"edge": "in"})]

        test_subset = [('a', 'b', {"edge": "in"})]

        sparse_subset = [('h', 'i', {"edge": "in"}),
                         ('i', 'j', {"edge": "in"}),
                         ('j', 'a', {"edge": "in"}),
                         ('a', 'c', {"edge": "in"}),
                         ('a', 'g', {"edge": "in"}),
                         ('b', 'c', {"edge": "in"}),
                         ('g', 'j', {"edge": "in"})]

        new_contradictions = [('b', 'c', {"edge": "in"}),
                              ('c', 'd', {"edge": "in"}),
                              ('d', 'e', {"edge": "in"}),
                              ('e', 'f', {"edge": "in"})]

        removed_contradictions = [('b', 'c', {"edge": "in"}),
                                  ('c', 'd', {"edge": "in"})]

        plot_graph_nice(GTest,
                        SAVE_DIR,
                        train_subset,
                        test_subset,
                        sparse_subset=sparse_subset,
                        new_contradictions=new_contradictions,
                        removed_contradictions=removed_contradictions)

        self.assertTrue(os.path.exists(f"{SAVE_DIR}/test_network_out.png"))
