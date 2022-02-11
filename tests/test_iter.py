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
