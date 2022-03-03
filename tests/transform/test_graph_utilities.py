"""Tests for manipulating the input data for kgemb-sens."""

# -*- coding: utf-8 -*-

import unittest

import networkx as nx
import numpy as np

from kgemb_sens.transform.graph_utilities import edge_dist, edge_degree, prob_dist, prob_dist_from_list, random_split_list, undirect_multidigraph
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
        cls.cc3.add_edges_from([(0, 1, {"edge": "blah"})])
        cls.cc3_degree_dict = dict(cls.cc3.degree())

        ### Nations
        ##cls.nations = load_benchmark_data_three_parts("nations", DATA_DIR)
        ##cls.assertEqual(cls.nations.number_of_edges(), 1992)

    def test_edge_dist(self):
        # Circular ladder
        # Optional To-Do: test edges not in graph. Now that it"s directed... could become dicey. Making assumptions
        # about when it is and isn"t directed sometimes.
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
        self.assertEqual(dist6, None)  # What behavior wanted?

        # Disconnected components graph
        dist7 = edge_dist((0, 1, {"edge": "test"}), (7, 8, {"edge": "test"}), self.cc3_dist_mat)
        self.assertEqual(dist7, float("inf"))

        # Test expected behavior for adjacent edges in KG
        dist8 = edge_dist((0, 1, {"edge": "test"}), (1, 0, {"edge": "test"}), self.cc3_dist_mat)
        self.assertEqual(dist8, 1)
        dist9 = edge_dist((0, 1, {"edge": "test"}), (0, 1, {"edge": "blah"}), self.cc3_dist_mat)
        self.assertEqual(dist9, 1)
        dist10 = edge_dist((0, 1, {"edge": "test"}), (0, 1, {"edge": "test"}), self.cc3_dist_mat)
        self.assertEqual(dist10, 0)

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

        # Distance-based calculations
        square_edge = (0, 1, 0, {"edge": "test"})
        cc3_pd1 = prob_dist(square_edge,
                            list(self.cc3.edges(data=True, keys=True)),
                            self.cc3_dist_mat,
                            self.cc3_degree_dict,
                            "distance",
                            self.cc3,
                            alpha=0)
        cc3_pd1 = dict(zip(list(self.cc3.edges(data="edge")), cc3_pd1))
        self.assertEqual(sum(cc3_pd1.values()), 1)
        self.assertEqual(np.count_nonzero(list(cc3_pd1.values())), 12)
        self.assertEqual(cc3_pd1[(0, 1, "test")], 0)

        triangle_edge_wrong = (5, 4, 0, {"edge": "Pizza"})
        triangle_edge = (5, 4, 0, {"edge": "test"})
        cc3_pd2_w = prob_dist(triangle_edge_wrong,
                              list(self.cc3.edges(data=True, keys=True)),
                              self.cc3_dist_mat,
                              self.cc3_degree_dict,
                              "distance",
                              self.cc3,
                              alpha=0)
        cc3_pd2_w = dict(zip(list(self.cc3.edges(data="edge")), cc3_pd2_w))
        self.assertAlmostEqual(cc3_pd2_w[(5, 4, "test")], 0.16667, places=3)
        self.assertEqual(np.count_nonzero(list(cc3_pd2_w.values())), 6)

        cc3_pd2 = prob_dist(triangle_edge,
                            list(self.cc3.edges(data=True, keys=True)),
                            self.cc3_dist_mat,
                            self.cc3_degree_dict,
                            "distance",
                            self.cc3,
                            alpha=0)
        cc3_pd2 = dict(zip(list(self.cc3.edges(data="edge")), cc3_pd2))
        self.assertEqual(cc3_pd2[(5, 4, "test")], 0)
        self.assertAlmostEqual(cc3_pd2[(5, 6, "test")], 0.2, places=3)

        def dist_alpha(in_alpha):
            p = prob_dist(("s", 0),  #NOTE: (0, "s") would fail #TODO: Do we care?
                          list(self.clg8.edges()),  # No keys or anything
                          self.clg8_dist_mat,
                          self.clg8_degree_dict,
                          "distance",
                          self.clg8,
                          alpha=in_alpha)
            p = dict(zip(list(self.clg8.edges(data="edge")), p))
            return p

        clg8_pd1, clg8_pd2, clg8_pd3, clg8_pd4, clg8_pd5 = list(map(dist_alpha, [0, 1, -1, -100, 100]))
        self.assertEqual(np.count_nonzero(list(clg8_pd1.values())), len(clg8_pd1.values())-1)
        self.assertEqual(len(set(clg8_pd1.values())), 2)  # All entries but the 0 are equal

        def get_max_prob_edges(in_dict):
            max_prob = max(in_dict.values())
            max_prob_edges = [edge for edge in in_dict.keys() if in_dict[edge] == max_prob]
            return max_prob_edges

        clg8_pd2_top_edges = get_max_prob_edges(clg8_pd2)
        self.assertTrue((4, "t", None) in clg8_pd2_top_edges)
        self.assertEqual(len(clg8_pd2_top_edges), 7)
        self.assertEqual(np.count_nonzero(list(clg8_pd2.values())), len(clg8_pd2.values())-1)

        clg8_pd3_top_edges = get_max_prob_edges(clg8_pd3)
        self.assertTrue((0, 7, None) in clg8_pd3_top_edges)
        self.assertEqual(len(clg8_pd3_top_edges), 6)
        self.assertEqual(np.count_nonzero(list(clg8_pd3.values())), len(clg8_pd3.values())-1)

        clg8_pd4_top_edges = get_max_prob_edges(clg8_pd4)
        self.assertTrue((0, 7, None) in clg8_pd4_top_edges)
        self.assertEqual(len(clg8_pd4_top_edges), 6)
        self.assertEqual(np.sum(np.array(list(clg8_pd4.values())) > 1e-10), 6)

        clg8_pd5_top_edges = get_max_prob_edges(clg8_pd5)
        self.assertTrue((4, "t", None) in clg8_pd5_top_edges)
        self.assertEqual(len(clg8_pd5_top_edges), 7)
        self.assertEqual(np.sum(np.array(list(clg8_pd5.values())) > 1e-10), 7)

        # Degree-based calculations
        square_edge = (0, 1, 0, {"edge": "test"})
        cc3_pd3 = prob_dist(square_edge,
                            list(self.cc3.edges(data=True, keys=True)),
                            self.cc3_dist_mat,
                            self.cc3_degree_dict,
                            "degree",
                            self.cc3,
                            alpha=-1)
        cc3_pd3 = dict(zip(list(self.cc3.edges(data="edge")), cc3_pd3))
        cc3_pd3_top_edges = get_max_prob_edges(cc3_pd3)
        self.assertAlmostEqual(sum(cc3_pd3.values()), 1)
        self.assertTrue((7, 8, "test") in cc3_pd3_top_edges)
        self.assertEqual(len(cc3_pd3_top_edges), 2)
        self.assertEqual(np.count_nonzero(list(cc3_pd3.values())), len(cc3_pd3.values())-1)
        self.assertTrue(cc3_pd3[(0, 1, "blah")] > 0)
        self.assertTrue(cc3_pd3[(1, 0, "test")] > 0)

        cc3_pd4 = prob_dist(square_edge,
                            list(self.cc3.edges(data=True, keys=True)),
                            self.cc3_dist_mat,
                            self.cc3_degree_dict,
                            "degree",
                            self.cc3,
                            alpha=1)
        cc3_pd4 = dict(zip(list(self.cc3.edges(data="edge")), cc3_pd4))
        cc3_pd4_top_edges = get_max_prob_edges(cc3_pd4)
        self.assertTrue((3, 2, "test") not in cc3_pd4_top_edges)
        self.assertTrue((1, 2, "test") in cc3_pd4_top_edges)
        self.assertTrue((0, 1, "blah") in cc3_pd4_top_edges)

        clg8_pd6 = prob_dist(("s", 0),  # NOTE: (0, "s") would fail #TODO: Do we care?
                             list(self.clg8.edges()),  # No keys or anything
                             self.clg8_dist_mat,
                             self.clg8_degree_dict,
                             "degree",
                             self.clg8,
                             alpha=100)
        clg8_pd6 = dict(zip(list(self.clg8.edges(data="edge")), clg8_pd6))
        clg8_pd6_top_edges = get_max_prob_edges(clg8_pd6)
        self.assertTrue((0, 7, None) in clg8_pd6_top_edges)
        self.assertTrue((7, 0, None) in clg8_pd6_top_edges)
        self.assertTrue((4, 5, None) in clg8_pd6_top_edges)
        self.assertEqual(clg8_pd6[("s", 0, None)], 0)
        self.assertEqual(len(clg8_pd6_top_edges), 12)
        self.assertEqual(np.sum(np.array(list(clg8_pd6.values())) > 1e-4), 12)

    def test_prob_dist_from_list(self):
        square_edge = (0, 1, 1, {"edge": "blah"})
        tri_edge = (4, 5, 0, {"edge": "test"})
        bi_edge = (7, 8, 0, {"edge": "test"})
        test_edges = [square_edge, tri_edge, bi_edge]
        #test_edges = [tri_edge]

        cc3_pdlist = prob_dist_from_list(test_edges,
                                         list(self.cc3.edges(data=True, keys=True)),
                                         self.cc3_dist_mat,
                                         self.cc3_degree_dict,
                                         "distance",
                                         self.cc3,
                                         alpha=1)
        print(cc3_pdlist)
        ##print(list(self.cc3.edges(data=True, keys=True)))
        cc3_pdlist = dict(zip(list(self.cc3.edges(data="edge")), cc3_pdlist))
        print(cc3_pdlist)
        self.assertEqual(np.count_nonzero(list(cc3_pdlist.values())), len(cc3_pdlist.values()) - 3)
        self.assertEqual(cc3_pdlist[(0, 1, "blah")], 0)
        self.assertEqual(cc3_pdlist[(4, 5, "test")], 0)
        self.assertNotAlmostEqual(cc3_pdlist[(5, 4, "test")], 0)
        self.assertEqual(cc3_pdlist[(7, 8, "test")], 0)

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
