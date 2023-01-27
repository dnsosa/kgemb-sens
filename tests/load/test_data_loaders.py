"""Tests for loading datasets for kgemb-sens."""

# -*- coding: utf-8 -*-

import unittest

from kgemb_sens.load.network_load import load_benchmark_data_three_parts, load_drkg_data
from kgemb_sens.analyze.metrics import calc_network_input_statistics

DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"


class TestDataLoaders(unittest.TestCase):
    """Tests for loading datasets for kgemb-sens."""

    def test_load_benchmark_data_three_parts(self):
        """Test that the benchmark datasets load as expected."""
        def get_all_rels(G):
            return set([r for _, _, r in G.edges(data='edge')])

        G_countries = load_benchmark_data_three_parts("countries", DATA_DIR)
        self.assertEqual(G_countries.number_of_edges(), 1159)
        self.assertEqual(G_countries.number_of_nodes(), 271)
        self.assertEqual(len(get_all_rels(G_countries)), 2)

        G_nations = load_benchmark_data_three_parts("nations", DATA_DIR)
        self.assertEqual(G_nations.number_of_edges(), 1992)
        self.assertEqual(G_nations.number_of_nodes(), 14)
        self.assertEqual(len(get_all_rels(G_nations)), 55)

        G_umls = load_benchmark_data_three_parts("umls", DATA_DIR)
        self.assertEqual(G_umls.number_of_edges(), 6529)
        self.assertEqual(G_umls.number_of_nodes(), 135)
        self.assertEqual(len(get_all_rels(G_umls)), 46)

        G_kinships = load_benchmark_data_three_parts("kinships", DATA_DIR)
        self.assertEqual(G_kinships.number_of_edges(), 10686)
        self.assertEqual(G_kinships.number_of_nodes(), 104)
        self.assertEqual(len(get_all_rels(G_kinships)), 25)

    @unittest.skip("Takes a while to load, don't need to test all the time")
    def test_load_drkg_data(self):
        """Test that the biological networks datasets load as expected."""

        G_gnbr_gg = load_drkg_data("gnbr_gg", DATA_DIR)
        self.assertEqual(G_gnbr_gg.number_of_edges(), 66722)
        G_gnbr_drdz = load_drkg_data("gnbr_drdz", DATA_DIR)
        self.assertEqual(G_gnbr_drdz.number_of_edges(), 77782)
        G_gnbr_drg = load_drkg_data("gnbr_drg", DATA_DIR)
        self.assertEqual(G_gnbr_drg.number_of_edges(), 80803)
        G_drugbank_drg = load_drkg_data("drugbank_drg", DATA_DIR)
        self.assertEqual(G_drugbank_drg.number_of_edges(), 24801)
        G_drugbank_drdz = load_drkg_data("drugbank_drdz", DATA_DIR)
        self.assertEqual(G_drugbank_drdz.number_of_edges(), 4968)

        G_gnbr_gg_pcnet = load_drkg_data("gnbr_gg", DATA_DIR, pcnet_filter=True, pcnet_dir=DATA_DIR)
        self.assertEqual(G_gnbr_gg_pcnet.number_of_edges(), 17048)

    def test_load_dengue_networks(self):
        G_gnbr_gg_dengue1 = load_drkg_data("gnbr_gg", DATA_DIR, dengue_filter=True)  # expand 1
        self.assertEqual(G_gnbr_gg_dengue1.number_of_edges(), 26)
        num_cc = calc_network_input_statistics(G_gnbr_gg_dengue1, calc_diam=True)[2]
        self.assertEqual(num_cc, 5)

        G_gnbr_gg_pcnet_dengue2 = load_drkg_data("gnbr_gg", DATA_DIR,
                                                 pcnet_filter=True, pcnet_dir=DATA_DIR,
                                                 dengue_filter=True, dengue_expand_depth=2)
        self.assertEqual(G_gnbr_gg_pcnet_dengue2.number_of_edges(), 94)
        med_rel_count = calc_network_input_statistics(G_gnbr_gg_pcnet_dengue2, calc_diam=True)[4]
        self.assertEqual(med_rel_count, 8)
