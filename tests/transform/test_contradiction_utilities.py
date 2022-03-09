"""Tests the helper functions for the contradictions pipeline."""

# -*- coding: utf-8 -*-

import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from collections import Counter

from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts
from kgemb_sens.transform.graph_utilities import edge_dist, undirect_multidigraph
from kgemb_sens.transform.contradiction_utilities import find_all_valid_negations, negative_completion,\
    generate_converse_edges_from, fill_with_contradictions, remove_contradictions

# from .resources.test_processing_pipeline_helpers import num_

DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"
SEED = 10


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
        cc3_nc = negative_completion(self.cc3, avn_cc3, ["test", "blah"], 1.0, SEED=SEED)
        clg8_nc = negative_completion(self.clg8, avn_clg8, ["test"], 0.2, SEED=SEED)
        rels_cc3_nc = set([r for _, _, r in cc3_nc.edges(data='edge')])

        # Test that the right predicates are added
        for rel in rels_cc3:
            self.assertTrue(rel in rels_cc3_nc)
            self.assertTrue(f"NOT-{rel}" in rels_cc3_nc)

        # Test expected number of edges
        self.assertEqual(cc3_nc.number_of_edges(), self.cc3.number_of_edges()*2)
        self.assertEqual(clg8_nc.number_of_edges(), round(self.clg8.number_of_edges()*1.2))

    def test_generate_converse_edges_from(self):
        e1 = (0, 10, None, {'edge': 'binds'})
        e2 = (5, 8, None, {'edge': 'NOT-binds'})
        e3 = (5, 8, None, {'edge': 'NOT-binds'})
        e4 = (5, "pizza", {'edge': 'treats'})
        e5 = ("rains", "evgr", {'distance': 50})
        e6 = ("trumpet", "guitar", "NOT-yes")
        e7 = ("maybe", "so")

        ne1 = (0, 10, None, {'edge': 'NOT-binds'})
        ne2 = (5, 8, None, {'edge': 'binds'})
        ne3 = (5, 8, None, {'edge': 'binds'})
        ne4 = (5, "pizza", {'edge': 'NOT-treats'})
        ne5 = ("rains", "evgr", {'distance': 50})
        ne6 = ("trumpet", "guitar", "yes")
        ne7 = ("maybe", "so")

        edge_list = [e1, e2, e3, e4, e5, e6, e7]
        nedge_list = [ne1, ne2, ne3, ne4, ne5, ne6, ne7]
        converse_edge_list1 = generate_converse_edges_from(edge_list)
        converse_edge_list2 = generate_converse_edges_from(nedge_list)
        converse_edge_list3 = generate_converse_edges_from(generate_converse_edges_from(edge_list))

        self.assertEqual(converse_edge_list1, nedge_list)
        self.assertEqual(converse_edge_list2, edge_list)
        self.assertEqual(converse_edge_list3, edge_list)

        e8 = (5, 8, 0, {'edge': 'NOT-binds'})
        ne8 = (5, 8, None, {'edge': 'binds'})
        self.assertEqual(generate_converse_edges_from([e8]), [ne8])

    def test_fill_with_contradictions(self):
        # DEGREE
        params = {'prob_type': "degree", 'alpha': 2, 'contradiction_frac': 0.5}
        edge_names = ["test", "blah"]
        val_test_subset = [(0, 1, 0, {'edge': 'test'})]
        G_contra, all_sampled_rel_edges, all_contradictory_edges = fill_with_contradictions(self.cc3, edge_names,
                                                                                            val_test_subset, params,
                                                                                            self.cc3_dist_mat,
                                                                                            self.cc3_degree_dict,
                                                                                            SEED=SEED)
        G_contra_rels_counter = Counter([r for _, _, r in G_contra.edges(data='edge')])
        self.assertEqual(G_contra.number_of_edges(), 32)
        self.assertEqual(G_contra_rels_counter["NOT-test"], 10)
        self.assertEqual(G_contra_rels_counter["NOT-blah"], 1)
        self.assertEqual(len(all_contradictory_edges), 11)
        self.assertEqual(len(all_sampled_rel_edges), 11)
        self.assertTrue(val_test_subset[0] not in all_sampled_rel_edges)

        # Check for them being sampled from correct component
        params = {'prob_type': "degree", 'alpha': -10, 'contradiction_frac': 0.2}
        edge_names = ["test"]
        val_test_subset = [(0, 1, 0, {'edge': 'test'})]
        G_contra, all_sampled_rel_edges, all_contradictory_edges = fill_with_contradictions(self.cc3, edge_names,
                                                                                            val_test_subset, params,
                                                                                            None,  # part of test
                                                                                            self.cc3_degree_dict,
                                                                                            SEED=SEED)
        G_contra_rels_counter = Counter([r for _, _, r in G_contra.edges(data='edge')])
        contra_edges_nodes = set([u for u, _, _, _ in all_contradictory_edges] + [v for _, v, _, _ in all_contradictory_edges])
        self.assertEqual(G_contra.number_of_edges(), 25)
        self.assertTrue("NOT-blah" not in G_contra_rels_counter.keys())
        self.assertEqual(G_contra_rels_counter["NOT-test"], 4)
        self.assertTrue(7 in contra_edges_nodes)  # with high probability

        # DISTANCE
        params = {'prob_type': "distance", 'alpha': 10, 'contradiction_frac': 0.2}
        edge_names = ["test"]
        val_test_subset = [("s", 0, 0, {'edge': 'test'})]
        G_contra, all_sampled_rel_edges, all_contradictory_edges = fill_with_contradictions(self.clg8, edge_names,
                                                                                            val_test_subset, params,
                                                                                            self.clg8_dist_mat,
                                                                                            None,  # part of test
                                                                                            SEED=SEED)
        G_contra_rels_counter = Counter([r for _, _, r in G_contra.edges(data='edge')])
        contra_edges_nodes = set([u for u, _, _, _ in all_contradictory_edges] + [v for _, v, _, _ in all_contradictory_edges])
        self.assertEqual(G_contra.number_of_edges(), 60)
        self.assertEqual(G_contra_rels_counter["NOT-test"], 10)
        self.assertTrue(4 in contra_edges_nodes)  # with high probability

        # What if had negative completed already?
        avn_clg8 = find_all_valid_negations(self.clg8)
        clg8_nc = negative_completion(self.clg8, avn_clg8, ["test"], 0.2, SEED=SEED)
        self.assertEqual(len(avn_clg8), 256)  # (18C2 * 2 - (24 * 2 + 2))
        self.assertEqual(clg8_nc.number_of_edges(), 60)  # 1.2 * 50 = 60

        params = {'prob_type': "distance", 'alpha': -1, 'contradiction_frac': 0.2}
        edge_names = set([r for _, _, r in clg8_nc.edges(data='edge')])
        val_test_subset = [(4, "t", 0, {'edge': 'test'})]
        clg8_nc_rels_counter = Counter([r for _, _, r in clg8_nc.edges(data='edge')])
        self.assertEqual(clg8_nc_rels_counter["NOT-test"], 10)

        G_contra, all_sampled_rel_edges, all_contradictory_edges = fill_with_contradictions(clg8_nc, edge_names,
                                                                                            val_test_subset, params,
                                                                                            self.clg8_dist_mat,
                                                                                            self.clg8_degree_dict,
                                                                                            SEED=SEED)
        print(G_contra.edges(data='edge', keys=True))
        G_contra_rels_counter = Counter([r for _, _, r in G_contra.edges(data='edge')])
        contra_edges_nodes = set([u for u, _, _, _ in all_contradictory_edges] + [v for _, v, _, _ in all_contradictory_edges])
        self.assertEqual(G_contra.number_of_edges(), 72)
        self.assertTrue("NOT-test" in G_contra_rels_counter.keys())
        self.assertEqual(G_contra_rels_counter["NOT-test"], 20)  # 10 + 10
        self.assertTrue(4 in contra_edges_nodes)  # with high probability
        edge_key_counter = Counter([k for _, _, k, _ in G_contra.edges(data='edge', keys=True)])
        self.assertEqual(edge_key_counter[1], 12)   # Count the number of contradictions (has a 1 key)

        # Now remove contradictions
        G_contra_remove, sampled_contras = remove_contradictions(G_contra, all_sampled_rel_edges,
                                                                 all_contradictory_edges, 0.5, SEED=SEED)
        edge_key_counter = Counter([k for _, _, k, _ in G_contra_remove.edges(data='edge', keys=True)])
        self.assertEqual(edge_key_counter[1], 6)   # Count the number of contradictions (has a 1 key)
        for edge in val_test_subset:
            self.assertTrue(edge not in sampled_contras)
        uv_set = set([(u, v) for u, v, _, _ in sampled_contras])
        self.assertEqual(len(uv_set), 6)

        # Finally one test with Nations
        params = {'prob_type': "degree", 'alpha': 2, 'contradiction_frac': 0.05}
        edge_names = ["exports3", "embassy", "accusation"]
        val_test_subset = [("netherlands", "uk", 0, {'edge': 'militaryalliance'})]
        G_contra, all_sampled_rel_edges, all_contradictory_edges = fill_with_contradictions(self.nations, edge_names,
                                                                                            val_test_subset, params,
                                                                                            None,
                                                                                            self.nations_degree_dict,
                                                                                            SEED=SEED)
        G_contra_rels_counter = Counter([r for _, _, r in G_contra.edges(data='edge')])
        self.assertEqual(G_contra.number_of_edges(), 2002)
        self.assertEqual(G_contra_rels_counter["NOT-exports3"], 2)
        self.assertEqual(G_contra_rels_counter["NOT-embassy"], 7)
        self.assertEqual(G_contra_rels_counter["NOT-accusation"], 1)
        self.assertEqual(len(set([r for _, _, r in G_contra.edges(data='edge')])), 58)
        self.assertEqual(len(all_contradictory_edges), 10)
        self.assertEqual(len(all_sampled_rel_edges), 10)
        self.assertTrue(val_test_subset[0] not in all_sampled_rel_edges)

        G_contra_remove, sampled_contras = remove_contradictions(G_contra, all_sampled_rel_edges,
                                                                 all_contradictory_edges, 1, SEED=SEED)
        edge_key_rels = set([rel for _, _, _, rel in G_contra_remove.edges(data='edge', keys=True)])

        self.assertFalse("NOT-exports3" in edge_key_rels)
        self.assertFalse("NOT-embassy" in edge_key_rels)
        self.assertFalse("NOT-accusation" in edge_key_rels)
