"""Tests for the embedding portion of kgemb-sens."""

# -*- coding: utf-8 -*-

import unittest

from kgemb_sens.load.data_loaders import load_benchmark_data_three_parts

DATA_DIR = "/Users/dnsosa/.data/pykeen/datasets"


class TestMetrics(unittest.TestCase):
    """Tests for calculating graph metrics for kgemb-sens."""

    @classmethod
    def setUpClass(cls):
        # Nations
        cls.nations = load_benchmark_data_three_parts("nations", DATA_DIR)

    def test_embed(self):
        # TODO
        # NOTE: is this redundant since it's being tested in the CLI?
        pass
