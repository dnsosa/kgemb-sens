# -*- coding: utf-8 -*-

"""Tests for ``kgemb_sens``."""

import unittest

from click.testing import CliRunner

from kgemb_sens import cli
from kgemb_sens.analyze import *
from kgemb_sens.experiment import *
from kgemb_sens.load import *

#from tests.constants import F1_PATH, F2_PATH


class TestKgembSens(unittest.TestCase):
    """Test KG embedding sensitivity experiments."""

    def test_kgemb_sens(self):
        """Test ``kgemb_sens``."""
        expected = None #fill in
        # actual = list(iter_together(F1_PATH, F2_PATH, sep=','))
        # self.assertEqual(expected, actual)
        pass  # TODO: Fill in

    def test_cli(self):
        """Test the ``iter_together`` command line interface."""
        #runner = CliRunner()
        # args = [F1_PATH, F2_PATH]
        # result = runner.invoke(cli.main, args)
        # self.assertEqual(0, result.exit_code)
        # expected_output = 'a,a_1,a_2\nb,b_1,b_2\nc,c_1,c_2\nd,d_1,d_2\n'
        # self.assertEqual(expected_output, result.output)
        pass  # TODO: Fill in


#GTest = nx.MultiDiGraph()
##GTest.add_edges_from([('a', 'b', {"edge": "in"}),
#                  ('b', 'c', {"edge": "in"}),
#                  ('c', 'd', {"edge": "in"}),
#                  ('d', 'e', {"edge": "in"}),
#                  ('e', 'f', {"edge": "in"}),
#                  ('f', 'g', {"edge": "in"}),
#                  ('g', 'h', {"edge": "in"}),
#                  ('h', 'i', {"edge": "in"}),
#                  ('i', 'j', {"edge": "in"}),
#                  ('j', 'a', {"edge": "in"}),
#                  ('a', 'c', {"edge": "in"}),
#                  ('a', 'g', {"edge": "in"}),
#                  ('b', 'c', {"edge": "under"}),
#                  ('b', 'c', {"edge": "over"}),
#                  #('b', 'c', {"edge": "around"}),
#                #('b', 'c', {"edge": "through"}),
#                  #('b', 'c', {"edge": "within"}),
#                  #('b', 'c', {"edge": "beside"}),
#                  ('g', 'j', {"edge": "in"})]
#                )
#
# train_subset = [('b', 'c', {"edge": "in"}),
#                   ('c', 'd', {"edge": "in"}),
#                   ('d', 'e', {"edge": "in"}),
#                   ('e', 'f', {"edge": "in"}),
#                   ('f', 'g', {"edge": "in"})]
# test_subset = [('a', 'b', {"edge": "in"})]
# sparse_subset = [('h', 'i', {"edge": "in"}),
#                   ('i', 'j', {"edge": "in"}),
#                   ('j', 'a', {"edge": "in"}),
#                   ('a', 'c', {"edge": "in"}),
#                   ('a', 'g', {"edge": "in"}),
#                   ('b', 'c', {"edge": "in"}),
#                   ('g', 'j', {"edge": "in"})]
# new_contradictions = [('b', 'c', {"edge": "in"}),
#                   ('c', 'd', {"edge": "in"}),
#                   ('d', 'e', {"edge": "in"}),
#                   ('e', 'f', {"edge": "in"})]
# removed_contradictions = [('b', 'c', {"edge": "in"}),
#                           ('c', 'd', {"edge": "in"})]

#
#
#
# FULL PIPELINE

# Input parameters

#SEED = 1005
#np.random.seed(SEED)

#params = {"dataset": "nations",
#          "val_test_frac": 1,
#          "val_fraction": 0,
#          "sparsified_frac": 0,
#          "alpha": -1,
#          "n_resample": 100,
#          "prob_type": "distance",  # distance, degree
#          "flatten_kg": "False",
#          "neg_completion_fraction": 0,
#          "contradiction_fraction": 0.1,
#          "contra_remove_fraction": 0,
#          "MODE": "contrasparsify",  # "sparsification", "contradictification", "contrasparsify"
#          "model_name": "transe",
#          "n_epochs": 200}
#
#G = load_data_three_parts(params["dataset"])
###G = GTest
#all_valid_negations = []
#
###edge_names = ["in"]
#if params["MODE"] in ["contradictification", "contrasparsify"]:
#    all_valid_negations, all_rels = find_all_valid_negations(G)
#
#all_results_list = []
#
#for i in range(params["n_resample"]):
#    data_paths, train_conditions_id, edge_divisions = graph_processing_pipeline(G, i, params, all_valid_negations,
#                                                                                all_rels, SEED)
#    train_subset, test_subset, sparse_subset, new_contradictions, removed_contradictions = edge_divisions
#    results_dict = run_embed_pipeline(data_paths, i, params, train_conditions_id)
#
#    all_results_list.append(results_dict)
#
#plot_graph_nice(GTest,
#                train_subset,
#                test_subset,
#                sparse_subset=sparse_subset,
#                new_contradictions=new_contradictions,
#                removed_contradictions=removed_contradictions)


#### nx.to_pandas_edgelist(G_out_train)