# -*- coding: utf-8 -*-

"""Tests for ``kgemb_sens``."""

import unittest
import os.path

import pandas as pd

from click.testing import CliRunner

from kgemb_sens import cli

# from tests.constants import TEST_DIR, DATA_DIR

OUT_DIR = "/Users/dnsosa/Desktop/AltmanLab/KGEmbSensitivity/kgemb-sens/tests/analyze/analyze_test_out"
DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"


class TestCli(unittest.TestCase):
    """Test KG embedding sensitivity experiments."""

    @unittest.skip("Takes a while to process Nations, don't need to test all the time")
    def test_cli(self):
        """Test the ``kgemb_sens`` command line interface."""
        runner = CliRunner()
        args = f""" --output_folder {OUT_DIR}
                    --data_dir {DATA_DIR}
                    --dataset nations
                    --n_resample 1
                    --n_epochs 1 
                    --neg_completion_frac 0
                    --contradiction_frac 1.0
                    --contra_remove_frac 0.5"""

        result = runner.invoke(cli.main, args)
        self.assertEqual(result.exit_code, 0)
        save_dir = "/Users/dnsosa/Desktop/AltmanLab/KGEmbSensitivity/kgemb-sens/tests/analyze/analyze_test_out/results/contrasparsify_alpha0.0_probtypedegree_flatFalse_sparsefrac0.0_negCompFrac0.0_contraFrac1.0_contraRemFrac0.5_vtfrac1.0_modeltranse"
        net_stats_path = os.path.join(save_dir, "network_stats.tsv")
        self.assertTrue(os.path.exists(net_stats_path))
        net_stats_df = pd.read_csv(net_stats_path, sep='\t')
        self.assertEqual(net_stats_df["n_ent_network"][0], 14)
        self.assertEqual(net_stats_df["post_n_rel_network"][0], 108)

        results_path = os.path.join(save_dir, "results.df")
        self.assertTrue(os.path.exists(results_path))
        results_df = pd.read_csv(results_path, sep='\t')
        self.assertEqual(results_df["Alpha"][0], 0)
        self.assertEqual(results_df["PCNet_filter"][0], False)
        self.assertEqual(results_df["MODE"][0], "contrasparsify")
        self.assertEqual(results_df["Run"][0], 0)
        self.assertTrue(results_df["Edge Degree"][0] > 0)

        head_preds = os.path.join(save_dir, "head_pred_run_0.tsv")
        self.assertTrue(os.path.exists(head_preds))
        tail_preds = os.path.join(save_dir, "tail_pred_run_0.tsv")
        self.assertTrue(os.path.exists(tail_preds))
